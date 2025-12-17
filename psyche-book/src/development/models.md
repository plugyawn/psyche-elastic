# Implementing models

This codebase includes a set of sample programs that let you design, implement, and test model architectures without spinning up the whole Psyche p2p training architecture.

We currently only implement Llama and Deepseek (see `shared/modeling/src/models/`), but PRs are very welcome to add more architectures and model types.

Both architectures support [MatFormer](../explain/matformer.md) for elastic inference, allowing clients with different hardware capabilities to participate in training at different capacity tiers.

The `train` example, documented below, is useful to test how your model trains using AdamW vs DisTrO.

## Running

```bash
cargo run --example train -- ---help
```

You'll need a pre-tokenized dataset downloaded to your disk for training.

> A PR is welcome to add an option to the trainer to use the HTTP data provider! You can refer to the http example in the data-provider crate for a sample implementation.

For a Llama 2 model, a pre-tokenized dataset to test with is available at [https://huggingface.co/datasets/emozilla/fineweb-10bt-tokenized-datatrove-llama2/](https://huggingface.co/datasets/emozilla/fineweb-10bt-tokenized-datatrove-llama2/tree/main).
Psyche only needs the `.ds` files, and will load any/all `.ds` files in the specified folder - you can download just one for smaller tests.

If you've downloaded part or all of the above dataset into a folder `data/fineweb-10bt` inside the Psyche repo, you can start a simple training run on a 20m parameter Llama 2 model:

```bash
cargo run --example train -- \
    --model emozilla/llama2-20m-init \
    --data-path ./data/fineweb-10bt/ \
    --total-batch 2 \
    --micro-batch 1
```

## Adding a new model type

The `train` example currently asssumes your model is a Llama or Deepseek v2/v3 model, and instantiates it via `(LlamaForCausalLM|DeepseekForCausalLM)::from_pretrained`.

We currently only support causal language models - to implement a new one, you can create a file similar to `llama_for_causal_lm` and implement your model, ensuring you provide a trait impl for `CausalLM`.

There's alpha-level support for models written in Python. See the [Python](./python.md) docs for more information.

You might also need to modify the data provider, if your data is structured in some way.
Since you're implementing the forward pass yourself, you can serve and interpret data passed from the data provider however you need.
The data provider currently only supports reading fixed-size batches from input files, so data batches with different sizes will require some additional work.

> PRs welcome for any new kinds of dataset loading!

## MatFormer Support

Models can implement MatFormer (Matryoshka Transformer) support to enable elastic inference. This allows the same model weights to be used at different capacity levels.

### Loading a Model with MatFormer Tier

```rust
use psyche_modeling::LlamaForCausalLM;

// Load at tier 1 (half FFN width)
let model = LlamaForCausalLM::from_pretrained_with_matformer_tier(
    &vs,
    repo_id,
    revision,
    device,
    1,  // matformer_tier
)?;
```

### Implementing MatFormer in a New Model

To add MatFormer support to a new model architecture:

1. **Store the tier configuration** in your model config struct:
   ```rust
   pub struct MyModelConfig {
       pub matformer_tier: u8,
       // ... other fields
   }
   ```

2. **Calculate the effective hidden size** based on tier:
   ```rust
   let matformer_hidden_size = match config.matformer_tier {
       0 => None,  // Full model
       tier => {
           let divisor = 1_i64.checked_shl(tier as u32)?;
           Some(config.intermediate_size / divisor)
       }
   };
   ```

3. **Use prefix slicing in forward pass**:
   ```rust
   fn forward(&self, xs: &Tensor) -> Tensor {
       if let Some(h) = self.matformer_hidden_size {
           // Use .narrow() to slice weight prefixes
           let w_up = self.up_proj.ws.narrow(0, 0, h);
           let w_down = self.down_proj.ws.narrow(1, 0, h);
           // ... rest of forward
       } else {
           // Full model path
       }
   }
   ```

4. **Add gradient isolation test**:
   ```rust
   #[test]
   fn matformer_has_zero_tail_grads() {
       // Verify exclusive weights have zero gradients
       // when training at smaller tiers
   }
   ```

See `shared/modeling/src/models/llama.rs` for a complete implementation example.
