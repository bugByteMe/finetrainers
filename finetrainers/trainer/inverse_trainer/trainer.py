import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import datasets.distributed
import torch
import wandb
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_layerwise_casting
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict
from tqdm import tqdm

from finetrainers import data, logging, models, optimizer, parallel, utils
from finetrainers.args import BaseArgsType
from finetrainers.config import TrainingType
from finetrainers.state import TrainState

from ..base import Trainer
from ..sft_trainer.config import SFTFullRankConfig, SFTLowRankConfig
from ..sft_trainer.trainer import SFTTrainer

from transformers import AutoTokenizer, UMT5EncoderModel

ArgsType = Union[BaseArgsType, SFTFullRankConfig, SFTLowRankConfig]

logger = logging.get_logger()

class EmbeddingWrap(torch.nn.Module):
    """
    A simple wrapper for the embedding to be trained.
    This is used to ensure that the embedding is treated as a single module
    and can be easily moved to the correct device.
    """

    def __init__(self, embedding: torch.nn.Module):
        # embedding: [B, L, Dim]
        super().__init__()
        self.scale = 1 # Scale the embedding's L dimension
        self.embedding = torch.nn.Parameter(embedding.clone().detach().repeat(1, self.scale, 1))
        self.init_embedding = embedding.clone().detach().repeat(1, self.scale, 1)
        self.init_embedding.requires_grad = False
        logger.info(f"Scale: {self.scale}, EmbeddingWrap scaled to shape: {self.embedding.shape}, requires_grad: {self.embedding.requires_grad}")

    def forward(self, *args, **kwargs):
        return self.embedding(*args, **kwargs)
    
    def loss(self) -> torch.Tensor:
        # Difference regularization loss
        return torch.norm(self.embedding - self.init_embedding.clone().detach(), p=2)

class InvTrainer(SFTTrainer):

    def __init__(self, args: ArgsType, model_specification: models.ModelSpecification) -> None:
        super().__init__(args, model_specification)
        self.embedding = None # Special embedding to be trained
        self.unwrap_transformer = None
        self.args.joint = True
        logger.info(f"Joint training: {self.args.joint}")
        torch.serialization.add_safe_globals([TrainState])
    
    def _prepare_models(self) -> None:
        logger.info("Initializing models")

        diffusion_components = self.model_specification.load_diffusion_models()
        self._set_components(diffusion_components)
        self.unwrap_transformer = self.transformer
        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )
        
    # Wrapper for prepare_conditions to handle the specific case of inverse training
    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }

        temp_tokens = tokenizer(
            caption,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        special_mask = temp_tokens.input_ids[0] == 900
        # Add batch dimension
        special_mask = special_mask.unsqueeze(0).to(dtype=torch.bool)
        logger.info(f"{temp_tokens.input_ids.shape}, {special_mask.shape} {special_mask.to(dtype=torch.bfloat16).sum()}")
        # logger.info(f"Special indices: {self.special_indices}, tokens: {temp_tokens.input_ids[0]}")
        conditions = self.model_specification.prepare_conditions(**conditions)
        conditions['special_mask'] = special_mask
        # logger.info(f"shape: {conditions['encoder_hidden_states'].shape}, dtype: {conditions['encoder_hidden_states'].dtype}")
        # conditions['encoder_hidden_states'][:, self.special_indices, :] = self.embedding.embedding
        # logger.info(f"Updated conditions with embedding: {conditions['encoder_hidden_states'].shape}")
        # logger.info(f"{self.embedding.embedding.requires_grad}, {conditions['encoder_hidden_states'].requires_grad}")
        return conditions
    
    def _prepare_checkpointing(self) -> None:
        parallel_backend = self.state.parallel_backend

        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.joint:
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # fmt: off
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self.args.target_modules,
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    # fmt: on
                    self.model_specification._save_lora_weights(
                        os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}"),
                        state_dict,
                        self.scheduler,
                        metadata,
                    )
                elif self.args.training_type == TrainingType.FULL_FINETUNE:
                    self.model_specification._save_model(
                        os.path.join(self.args.output_dir, "model_weights", f"{self.state.train_state.step:06d}"),
                        self.transformer,
                        state_dict,
                        self.scheduler,
                    )
            parallel_backend.wait_for_everyone()

        enable_state_checkpointing = self.args.checkpointing_steps > 0
        self.checkpointer = parallel_backend.get_checkpointer(
            dataloader=self.dataloader,
            model_parts=[self.embedding] if not self.args.joint else [self.embedding, self.transformer],
            optimizers=self.optimizer,
            schedulers=self.lr_scheduler,
            states={"train_state": self.state.train_state},
            checkpointing_steps=self.args.checkpointing_steps,
            checkpointing_limit=self.args.checkpointing_limit,
            output_dir=self.args.output_dir,
            enable=enable_state_checkpointing,
            _callback_fn=save_model_hook,
        )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            self.checkpointer.load(resume_from_checkpoint)
        
        self.state.train_state = self.checkpointer.states['train_state'] # TODO??

        # Load LoRA weights, only for DB and IV post training
        # from diffusers import WanPipeline
        # from diffusers.loaders import WanLoraLoaderMixin
        # lora_cls = WanLoraLoaderMixin()
        # kwargs = {}
        # kwargs["return_lora_metadata"] = True
        # state_dict, metadata = lora_cls.lora_state_dict("/home/jrguo/finetrainers/demo2/output_lora_sled-dog-2000/lora_weights/002000", **kwargs)
        # logger.info(f"Model dict: {list(self.unwrap_transformer.named_modules())}")
        # logger.info(f"State dict: {list(state_dict.keys())}")
        # lora_cls.load_lora_into_transformer(
        #     state_dict,
        #     transformer=self.unwrap_transformer,
        #     adapter_name="wan-lora",
        #     metadata=metadata,
        #     _pipeline=None,
        #     low_cpu_mem_usage=False,
        #     hotswap=False,
        # )
        if self.args.gradient_checkpointing:
            # TODO(aryan): support other checkpointing types
            utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")
        
        # print(self.unwrap_transformer.state_dict().keys())
        # self.unwrap_transformer.load_lora_weights("/home/jrguo/finetrainers/demo2/output_lora_sled-dog-2000/lora_weights/002000", adapter_name="wan-lora")
        # self.unwrap_transformer.set_adapters(["wan-lora"], [1.0])
        # pipeline_cls = WanPipeline(self.tokenizer, self.text_encoder, self.unwrap_transformer, self.vae, self.scheduler)
        # pipeline_cls.load_lora_weights("/home/jrguo/finetrainers/demo2/output_lora_sled-dog-2000/lora_weights/002000", adapter_name="wan-lora")
        # pipeline_cls.set_adapters(["wan-lora"], [1.0])
        # self.checkpointer.save(
        #     step=0, _device=self.state.parallel_backend.device, _is_main_process=parallel_backend.is_main_process
        # )
        # exit(0)

    def _prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        parallel_backend = self.state.parallel_backend

        # Load tokenizers and text encoders
        condition_components = self.model_specification.load_condition_models()
        component_names = list(condition_components.keys())
        component_modules = list(condition_components.values())
        logger.info(f"Loaded condition components: {component_names}")
        self._set_components(condition_components)
        self._move_components_to_device(component_modules)
        self._maybe_torch_compile()
        
        # Load potential checkpoints
        # items = os.listdir(self.args.output_dir)
        # items = [item for item in items if item.endswith(".pt")]
        # items.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        # logger.info(f"Found {len(items)} potential checkpoints in {self.args.output_dir}")
        # if items:
        #     latest_checkpoint = items[-1]
        #     logger.info(f"Loading latest checkpoint: {latest_checkpoint}")
        #     checkpoint_path = os.path.join(self.args.output_dir, latest_checkpoint)
        #     self.embedding = torch.load(checkpoint_path, map_location=parallel_backend.device)
        # else:
        embedding = self.model_specification.prepare_conditions(self.tokenizer, self.text_encoder, os.environ["INIT_TOKEN"])['encoder_hidden_states'][:, :1, :]
        
        logger.info(f"Embedding shape: {embedding.shape}")
        logger.info("Finetuning textual embedding parameters")
        self.embedding = EmbeddingWrap(embedding)

        utils.set_requires_grad([self.transformer], False)
        utils.set_requires_grad([self.embedding], True)
        logger.info(f"embedding requires_grad: {self.embedding.requires_grad_}, {self.embedding.embedding.requires_grad}")
        self._delete_components(component_names)
        del condition_components, component_names, component_modules

        assert self.args.training_type == "inv"

        if self.args.joint:
            transformer_lora_config = None
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
        # Make sure the trainable params are in float32 if data sharding is not enabled. For FSDP, we need all
        # parameters to be of the same dtype.
        if parallel_backend.data_sharding_enabled:
            self.transformer.to(dtype=self.args.transformer_dtype)
        else:
            if self.args.joint:
                cast_training_params([self.transformer], dtype=torch.float32)
    
    # Modify trainable parameters to include the embedding
    def _prepare_for_training(self) -> None:
        # 1. Apply parallelism
        parallel_backend = self.state.parallel_backend
        model_specification = self.model_specification

        if parallel_backend.context_parallel_enabled:
            parallel_backend.apply_context_parallel(self.transformer, parallel_backend.get_mesh()["cp"])
            parallel_backend.apply_context_parallel(self.embedding, parallel_backend.get_mesh()["cp"])

        if parallel_backend.tensor_parallel_enabled:
            # TODO(aryan): handle fp8 from TorchAO here
            model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh()["tp"],
                transformer=self.transformer,
            )

        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            # TODO(aryan): support other checkpointing types
            # Dealyed to checkpointing setup
            # utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")
            utils.apply_activation_checkpointing(self.embedding, checkpointing_type="full")

        # Apply torch.compile
        self._maybe_torch_compile()

        # Enable DDP, FSDP or HSDP
        if parallel_backend.data_sharding_enabled:
            # TODO(aryan): remove this when supported
            if self.args.parallel_backend == "accelerate":
                raise NotImplementedError("Data sharding is not supported with Accelerate yet.")

            dp_method = "HSDP" if parallel_backend.data_replication_enabled else "FSDP"
            logger.info(f"Applying {dp_method} on the model")

            if parallel_backend.data_replication_enabled or parallel_backend.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            parallel_backend.apply_fsdp2(
                model=self.transformer,
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,  # TODO(aryan): needs to be tested and allowed for enabling later
                device_mesh=parallel_backend.get_mesh()[dp_mesh_names],
            )
        elif parallel_backend.data_replication_enabled:
            if parallel_backend.get_mesh().ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")
            logger.info("Applying DDP to the model")
            parallel_backend.apply_ddp(self.transformer, parallel_backend.get_mesh())
        else:
            if self.args.joint:
                parallel_backend.prepare_model(self.transformer)
            parallel_backend.prepare_model(self.embedding)
            logger.info(f"Prepare Model!!!")

        self._move_components_to_device()

        # 2. Prepare optimizer and lr scheduler
        # For training LoRAs, we can be a little more optimal. Currently, the OptimizerWrapper only accepts torch::nn::Module.
        # This causes us to loop over all the parameters (even ones that don't require gradients, as in LoRA) at each optimizer
        # step. This is OK (see https://github.com/pytorch/pytorch/blob/2f40f789dafeaa62c4e4b90dbf4a900ff6da2ca4/torch/optim/sgd.py#L85-L99)
        # but can be optimized a bit by maybe creating a simple wrapper module encompassing the actual parameters that require
        # gradients. TODO(aryan): look into it in the future.
        model_parts = [self.embedding] if not self.args.joint else [self.embedding, self.transformer]
        self.state.num_trainable_parameters = sum(
            p.numel() for m in model_parts for p in m.parameters() if p.requires_grad
        )

        # Setup distributed optimizer and lr scheduler
        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_state = TrainState()
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
            # TODO(aryan): handle last_epoch
        )
        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(self.optimizer, self.lr_scheduler)

        # 3. Initialize trackers, directories and repositories
        self._init_logging()
        self._init_trackers()
        self._init_directories_and_repositories()

    def _save_embedding(self, step: int) -> None:
        """
        Save the embedding to the output directory.
        This is used to save the embedding at regular intervals during training.
        """
        embedding_path = os.path.join(self.args.output_dir, f"embedding_step_{step}.pt")
        torch.save(self.embedding.state_dict(), embedding_path)
        logger.info(f"Saved embedding to {embedding_path}")

    def _train(self) -> None:
        logger.info("Starting training")

        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        device = parallel_backend.device
        dtype = self.args.transformer_dtype

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        global_batch_size = self.args.batch_size * parallel_backend._dp_degree
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-replica batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=train_state.step,
            desc="Training steps",
            disable=not parallel_backend.is_local_main_process,
        )

        generator = torch.Generator(device=device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        scheduler_sigmas = utils.get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=device, dtype=torch.float32) if scheduler_sigmas is not None else None
        )
        scheduler_alphas = utils.get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=device, dtype=torch.float32) if scheduler_alphas is not None else None
        )
        # timesteps_buffer = []

        if self.args.joint:
            self.transformer.train()
        self.embedding.train()
        data_iterator = iter(self.dataloader)

        compute_posterior = False if self.args.enable_precomputation else (not self.args.precomputation_once)
        preprocessor = data.initialize_preprocessor(
            rank=parallel_backend.rank,
            world_size=parallel_backend.world_size,
            num_items=self.args.precomputation_items if self.args.enable_precomputation else 1,
            processor_fn={
                "condition": self.prepare_conditions,
                "latent": functools.partial(
                    self.model_specification.prepare_latents, compute_posterior=compute_posterior
                ),
            },
            save_dir=self.args.precomputation_dir,
            enable_precomputation=self.args.enable_precomputation,
            enable_reuse=self.args.precomputation_reuse,
        )
        condition_iterator: Iterable[Dict[str, Any]] = None
        latent_iterator: Iterable[Dict[str, Any]] = None
        sampler = data.ResolutionSampler(
            batch_size=self.args.batch_size, dim_keys=self.model_specification._resolution_dim_keys
        )
        requires_gradient_step = True
        accumulated_loss = 0.0

        while (
            train_state.step < self.args.train_steps and train_state.observed_data_samples < self.args.max_data_samples
        ):
            # 1. Load & preprocess data if required
            if preprocessor.requires_data:
                condition_iterator, latent_iterator = self._prepare_data(preprocessor, data_iterator)

            # 2. Prepare batch
            with self.tracker.timed("timing/batch_preparation"):
                try:
                    condition_item = next(condition_iterator)
                    latent_item = next(latent_iterator)
                    sampler.consume(condition_item, latent_item)
                except StopIteration:
                    if requires_gradient_step:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        requires_gradient_step = False
                    logger.info("Data exhausted. Exiting training loop.")
                    break

                if sampler.is_ready:
                    condition_batch, latent_batch = sampler.get_batch()
                    condition_model_conditions = self.model_specification.collate_conditions(condition_batch)
                    latent_model_conditions = self.model_specification.collate_latents(latent_batch)
                else:
                    continue

            train_state.step += 1
            train_state.observed_data_samples += self.args.batch_size * parallel_backend._dp_degree

            logger.debug(f"Starting training step ({train_state.step}/{self.args.train_steps})")

            latent_model_conditions = utils.align_device_and_dtype(latent_model_conditions, device, dtype)
            condition_model_conditions = utils.align_device_and_dtype(condition_model_conditions, device, dtype)
            latent_model_conditions = utils.make_contiguous(latent_model_conditions)
            condition_model_conditions = utils.make_contiguous(condition_model_conditions)

            # 2.5 Replace embeddings
            # logger.info(condition_model_conditions.keys())
            # logger.info(f"{condition_model_conditions['encoder_hidden_states'].shape}, {condition_model_conditions['encoder_hidden_states'].requires_grad}")
            # convert mask from float to bool
            condition_model_conditions['special_mask'] = condition_model_conditions['special_mask'].to(torch.bool)
            # logger.info(f"Embedding before replacement: {condition_model_conditions['encoder_hidden_states'].shape}, requires_grad: {condition_model_conditions['encoder_hidden_states'].requires_grad}")
            for B in range(condition_model_conditions['encoder_hidden_states'].shape[0]):
                for L in range(condition_model_conditions['encoder_hidden_states'].shape[1]):
                    if condition_model_conditions['special_mask'][B, L]:
                        # logger.info(f"Replacing embedding for batch {B}, token {L}")
                        condition_model_conditions['encoder_hidden_states'][B] = \
                            torch.cat([condition_model_conditions['encoder_hidden_states'][B, :L, :], 
                                      self.embedding.embedding[0, :],
                                      condition_model_conditions['encoder_hidden_states'][B, L + 1:, :]], dim=0)[:512] # TODO: max sequence length
            # logger.info(f"Embedding after replacement: {condition_model_conditions['encoder_hidden_states'].shape}, requires_grad: {condition_model_conditions['encoder_hidden_states'].requires_grad}")
            # logger.info(f"mask dtype: {condition_model_conditions['special_mask']}, shape: {condition_model_conditions['special_mask'].shape}")
            # condition_model_conditions['encoder_hidden_states'][condition_model_conditions['special_mask'], :] = self.embedding.embedding
            # logger.info(f"{condition_model_conditions['encoder_hidden_states'].shape}, {condition_model_conditions['encoder_hidden_states'].requires_grad}")
            condition_model_conditions.pop("special_mask", None)

            # 3. Forward pass
            sigmas = utils.prepare_sigmas(
                scheduler=self.scheduler,
                sigmas=scheduler_sigmas,
                batch_size=self.args.batch_size,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
                flow_logit_mean=self.args.flow_logit_mean,
                flow_logit_std=self.args.flow_logit_std,
                flow_mode_scale=self.args.flow_mode_scale,
                device=device,
                generator=self.state.generator,
            )
            sigmas = utils.expand_tensor_dims(sigmas, latent_model_conditions["latents"].ndim)

            # NOTE: for planned refactor, make sure that forward and backward pass run under the context.
            # If only forward runs under context, backward will most likely fail when using activation checkpointing
            with self.attention_provider_ctx(training=True):
                with self.tracker.timed("timing/forward"):
                    pred, target, sigmas = self.model_specification.forward(
                        transformer=self.transformer,
                        scheduler=self.scheduler,
                        condition_model_conditions=condition_model_conditions,
                        latent_model_conditions=latent_model_conditions,
                        sigmas=sigmas,
                        compute_posterior=compute_posterior,
                    )

                timesteps = (sigmas * 1000.0).long()
                weights = utils.prepare_loss_weights(
                    scheduler=self.scheduler,
                    alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                    sigmas=sigmas,
                    flow_weighting_scheme=self.args.flow_weighting_scheme,
                )
                weights = utils.expand_tensor_dims(weights, pred.ndim)

                # 4. Compute loss & backward pass
                with self.tracker.timed("timing/backward"):
                    loss = weights.float() * (pred.float() - target.float()).pow(2)
                    # Average loss across all but batch dimension (for per-batch debugging in case needed)
                    loss = loss.mean(list(range(1, loss.ndim)))
                    # Average loss across batch dimension
                    loss = loss.mean()
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    logger.info(f"loss requires_grad: {loss.requires_grad}, loss: {loss.item()}")
                    # Regularization loss
                    reg_loss = self.embedding.loss()
                    if self.args.gradient_accumulation_steps > 1:
                        reg_loss = reg_loss / self.args.gradient_accumulation_steps
                    logger.info(f"reg_loss requires_grad: {reg_loss.requires_grad}, reg_loss: {reg_loss.item()}")
                    # loss += 0.3 * reg_loss
                    loss.backward()
                    logger.info(f"Embedding grad norm: {self.embedding.embedding.grad.norm()}")
                accumulated_loss += loss.detach().item()
                requires_gradient_step = True

            # 5. Clip gradients
            model_parts = [self.embedding] if not self.args.joint else [self.embedding, self.transformer]
            grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                self.args.max_grad_norm,
                foreach=True,
                pp_mesh=parallel_backend.get_mesh()["pp"] if parallel_backend.pipeline_parallel_enabled else None,
            )

            # 6. Step optimizer & log metrics
            logs = {}
            
            if train_state.step % self.args.gradient_accumulation_steps == 0:
                # TODO(aryan): revisit no_sync() for FSDP
                initial_embedding = self.embedding.embedding.clone().detach()
                with self.tracker.timed("timing/optimizer_step"):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                logger.info(f"embedding: {self.embedding.embedding.sum()}")
                logger.info(f"embedding change: {(initial_embedding - self.embedding.embedding).norm()}")
                logs["train/lr"] = self.lr_scheduler.get_last_lr()[0] if isinstance(self.lr_scheduler, dict) else self.lr_scheduler.get_last_lr()[0]
                logs["train/reg_loss"] = reg_loss.detach().item() * self.args.gradient_accumulation_steps
                if grad_norm is not None:
                    grad_norm = grad_norm if isinstance(grad_norm, float) else grad_norm.detach().item()
                if (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.data_sharding_enabled
                    or parallel_backend.context_parallel_enabled
                ):
                    dp_cp_mesh = parallel_backend.get_mesh()["dp_cp"]
                    if grad_norm is not None:
                        grad_norm = parallel.dist_mean(torch.tensor([grad_norm], device=device), dp_cp_mesh)
                    global_avg_loss, global_max_loss = (
                        parallel.dist_mean(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                        parallel.dist_max(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                    )
                else:
                    global_avg_loss = global_max_loss = accumulated_loss

                logs["train/global_avg_loss"] = global_avg_loss
                logs["train/global_max_loss"] = global_max_loss
                if grad_norm is not None:
                    logs["train/grad_norm"] = grad_norm
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)
                accumulated_loss = 0.0
                requires_gradient_step = False

            progress_bar.update(1)
            progress_bar.set_postfix(logs)

            # timesteps_buffer.extend([(train_state.step, t) for t in timesteps.detach().cpu().numpy().tolist()])

            if train_state.step % self.args.logging_steps == 0:
                # TODO(aryan): handle non-SchedulerWrapper schedulers (probably not required eventually) since they might not be dicts
                # TODO(aryan): causes NCCL hang for some reason. look into later
                # logs.update(self.lr_scheduler.get_last_lr())

                # timesteps_table = wandb.Table(data=timesteps_buffer, columns=["step", "timesteps"])
                # logs["timesteps"] = wandb.plot.scatter(
                #     timesteps_table, "step", "timesteps", title="Timesteps distribution"
                # )
                # timesteps_buffer = []

                logs["train/observed_data_samples"] = train_state.observed_data_samples

                parallel_backend.log(logs, step=train_state.step)
                train_state.log_steps.append(train_state.step)

            # 7. Save checkpoint if required # TODO
            with self.tracker.timed("timing/checkpoint"):
                if train_state.step % self.args.checkpointing_steps == 0:
                    self.checkpointer.save(
                        step=train_state.step, _device=device, _is_main_process=parallel_backend.is_main_process
                    )
                    self._save_embedding(train_state.step)

            # if train_state.step % 30 == 0:
            #     self._validate(step=train_state.step, final_validation=False)
                # exit(0)
            # 8. Perform validation if required # TODO
            if train_state.step % self.args.validation_steps == 0:
                self._validate(step=train_state.step, final_validation=False)

        # 9. Final checkpoint, validation & cleanup
        self.checkpointer.save(
            train_state.step, force=True, _device=device, _is_main_process=parallel_backend.is_main_process
        )
        parallel_backend.wait_for_everyone()
        self._validate(step=train_state.step, final_validation=True)

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        # 10. Upload artifacts to hub
        if parallel_backend.is_main_process and self.args.push_to_hub:
            upload_folder(
                repo_id=self.state.repo_id,
                folder_path=self.args.output_dir,
                ignore_patterns=[f"{self.checkpointer._prefix}_*"],
            )

        parallel_backend.destroy()
    
    def _validate(self, step: int, final_validation: bool = False) -> None:
        if self.args.validation_dataset_file is None:
            return

        logger.info("Starting validation")

        # 1. Load validation dataset
        parallel_backend = self.state.parallel_backend
        dataset = data.ValidationDataset(self.args.validation_dataset_file)

        # Hack to make accelerate work. TODO(aryan): refactor
        if parallel_backend._dp_degree > 1:
            dp_mesh = parallel_backend.get_mesh()["dp"]
            dp_local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
            dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, dp_local_rank, dp_world_size)
        else:
            dp_mesh = None
            dp_local_rank, dp_world_size = parallel_backend.local_rank, 1

        validation_dataloader = data.DPDataLoader(
            dp_local_rank,
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items,
        )
        data_iterator = iter(validation_dataloader)
        main_process_prompts_to_filenames = {}  # Used to save model card
        all_processes_artifacts = []  # Used to gather artifacts from all processes

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)
        pipeline = self._init_pipeline(final_validation=final_validation)

        # 2. Run validation
        # TODO(aryan): when running validation with FSDP, if the number of data points is not divisible by dp_shards, we
        # will hang indefinitely. Either pad the dataset or raise an error early on during initialization if the dataset
        # size is not divisible by dp_shards.
        if self.args.joint:
            self.transformer.eval()
            # self.transformer.requires_grad_(False)
        self.embedding.eval()
        # self.embedding.requires_grad_(False)
        while True:
            validation_data = next(data_iterator, None)
            if validation_data is None:
                break

            validation_data = validation_data[0]
            with self.attention_provider_ctx(training=False):
                validation_artifacts = self.model_specification.validation(
                    pipeline=pipeline, generator=generator, special_embedding=self.embedding.embedding.clone().detach(), **validation_data
                )

            if dp_local_rank != 0:
                continue

            PROMPT = validation_data["prompt"]
            IMAGE = validation_data.get("image", None)
            VIDEO = validation_data.get("video", None)
            EXPORT_FPS = validation_data.get("export_fps", 30)

            # 2.1. If there are any initial images or videos, they will be logged to keep track of them as
            # conditioning for generation.
            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            artifacts = {
                "input_image": data.ImageArtifact(value=IMAGE),
                "input_video": data.VideoArtifact(value=VIDEO),
            }

            # 2.2. Track the artifacts generated from validation
            for i, validation_artifact in enumerate(validation_artifacts):
                if validation_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": validation_artifact})

            # 2.3. Save the artifacts to the output directory and create appropriate logging objects
            # TODO(aryan): Currently, we only support WandB so we've hardcoded it here. Needs to be revisited.
            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact))
                if artifact.value is None:
                    continue

                time_, rank, ext = int(time.time()), parallel_backend.rank, artifact.file_extension
                filename = "validation-" if not final_validation else "final-"
                filename += f"{step}-{rank}-{index}-{prompt_filename}-{time_}.{ext}"
                output_filename = os.path.join(self.args.output_dir, filename)

                if parallel_backend.is_main_process and ext in ["mp4", "jpg", "jpeg", "png"]:
                    main_process_prompts_to_filenames[PROMPT] = filename

                if isinstance(artifact, data.ImageArtifact):
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif isinstance(artifact, data.VideoArtifact):
                    export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))

        # 3. Cleanup & log artifacts
        parallel_backend.wait_for_everyone()

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "image_processor", "vae"]
        if self.args.enable_precomputation:
            self._delete_components(module_names)
        torch.cuda.reset_peak_memory_stats(parallel_backend.device)

        # Gather artifacts from all processes. We also need to flatten them since each process returns a list of artifacts.
        all_artifacts = [None] * dp_world_size
        if dp_world_size > 1:
            torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        else:
            all_artifacts = [all_processes_artifacts]
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_backend.is_main_process:
            tracker_key = "final" if final_validation else "validation"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_backend.log({tracker_key: artifact_log_dict}, step=step)

            if self.args.push_to_hub and final_validation:
                video_filenames = list(main_process_prompts_to_filenames.values())
                prompts = list(main_process_prompts_to_filenames.keys())
                utils.save_model_card(
                    args=self.args, repo_id=self.state.repo_id, videos=video_filenames, validation_prompts=prompts
                )

        parallel_backend.wait_for_everyone()
        if not final_validation:
            self._move_components_to_device()
            # self.embedding.requires_grad_(True)
            self.embedding.train()
            if self.args.joint:
                # self.transformer.requires_grad_(True)
                self.transformer.train()