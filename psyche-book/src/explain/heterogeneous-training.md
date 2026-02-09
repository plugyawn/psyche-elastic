# Heterogeneous Training in Psyche

Here is a constraint: not everyone has the same hardware.

Until now, "training a model" meant that every participant runs the full model – same architecture, same width, same memory footprint. But what if we relaxed that? What if "training" meant something more flexible – participants contributing gradients to a shared model, even if they can only run a *subset* of that model?

This is heterogeneous training. And it rests on a structure called MatFormer.

## The deployment problem MatFormer solves

Take a standard transformer. You train it once, and you get one model. If you want a faster version, you train a separate, smaller model. Want a mobile version? Train another one. Each variant needs its own training run, its own hyperparameter sweep, its own validation.

MatFormer's insight: what if you could train *once* and extract multiple models of different sizes – all sharing weights, all behaviorally consistent?

The trick is in the FFN blocks. These are the workhorses of the transformer – typically 2/3 of the parameters. A standard FFN looks like:

$$\text{FFN}(x) = W_{\text{down}} \cdot \sigma(W_{\text{up}} \cdot x)$$

Now here's the move: instead of always using all neurons in the hidden layer, what if you sometimes used only the first half? And sometimes only the first quarter?

As you can see, you get a smaller model. Not very interesting by itself. But note that this smaller model shares its weights with the larger one – the first 512 neurons are identical in both. That is the consequence of prefix slicing.

What *is* interesting is if you tweak the training objective. Go from "always train the full model" to "randomly sample a width, and train only that slice."

If the first 512 neurons must sometimes function as a complete model by themselves, they are forced to carry useful information independently of the remaining neurons.

The early neurons become *more important*. They must solve the task alone when sampled at smaller widths, so they encode the most critical features. The later neurons learn refinements – corrections that improve upon what the prefix already does.

## Breaking permutation symmetry

What is happening here? How did we go from interchangeable neurons to an implicit ordering?

In a vanilla FFN, hidden neurons are exchangeable. There is nothing intrinsically special about "neuron 47" versus "neuron 891" – they are permutation invariant. Swap any two neurons (with corresponding swaps in the weight matrices), and the function is identical.

The choice of "first m neurons" for the smaller model is arbitrary. Any m neurons would work equally well – if you only ever trained the full model.

What breaks the symmetry is the training objective itself.

Once you add terms like $L(M_i(x), y)$ where $M_i$ uses only indices $[0, m_i)$, the loss is no longer permutation-invariant. Swapping neuron 0 with neuron $h-1$ changes which subnetworks contain each neuron. Position now determines participation in the loss landscape.

This is the same identifiability argument as *nested dropout* (Rippel et al., 2014), which recovers PCA-like orderings in autoencoders by training with random truncation. It is the same mechanism underlying *Matryoshka Representation Learning*, where embedding prefixes are trained to be useful independently.

Think Matryoshka dolls – Russian nesting dolls where each smaller doll is complete in itself. With a simple change to the training rule, we go from homogeneous neurons to a hierarchy.

## Why prefix neurons stabilize (three mechanisms)

A reasonable objection: "Isn't this multi-objective optimization? Won't the small model and the large model fight each other?"

They would, if they were unrelated objectives. But the relationship here is nested, and that changes everything.

**Mechanism 1: The loss directly penalizes breaking the small model.**

When granularity $i$ is sampled, only $M_i$ participates in the forward pass. Any update that degrades $M_i$'s performance is punished by the loss $L(M_i(x), y)$.

Formally: let $\theta_{\text{prefix}}$ denote weights in $[0, m_1)$. These weights are in the computational graph for *all* granularities. An update $\Delta\theta_{\text{prefix}}$ that hurts $M_1$ will be punished whenever $i=1$ is sampled.

A common misunderstanding is that the prefix is "free to drift" between sampling events. It is not – the small model's loss term is always watching.

**Mechanism 2: Shared neurons get more gradient updates.**

With $g$ granularities and uniform sampling:

| Neuron Range | Active in Granularities | Gradient Frequency |
|--------------|------------------------|-------------------|
| $[0, m_1)$ | All $g$ | 100% |
| $[m_1, m_2)$ | $\{2, \ldots, g\}$ | $(g-1)/g$ |
| $[m_{g-1}, m_g)$ | $\{g\}$ only | $1/g$ |

This is a simple counting argument from the training rule. Prefix neurons receive $g\times$ more gradient updates than suffix neurons.

**Mechanism 3: Larger widths learn residual corrections.**

Decompose the full model output:

$$M_g(x) = M_1(x) + \underbrace{\bigl(M_g(x) - M_1(x)\bigr)}_{\text{suffix contribution}}$$

If $M_1$ is trained to be good alone, the suffix learns a *residual correction* – what can be improved given that the prefix already solved most of the problem.

The objectives are not adversarial. Larger models have strictly more capacity. If $M_1$ converges to some $f^* \in \mathcal{F}_1$, then $M_g$ can represent $f^*$ exactly (using suffix weights of zero) or improve upon it. The extra capacity is a refinement channel, not a competing objective.

## The residual framing (what made it click for me)

The nested function class argument ($\mathcal{F}_1 \subset \mathcal{F}_2 \subset \cdots \subset \mathcal{F}_g$) is correct, but somewhat abstract. Here is the intuition that made it tangible for me.

Think of it like a residual network. The suffix computes:

$$\text{suffix contribution} = M_g(x) - M_1(x)$$

When we sample granularity 1, only $M_1$ trains. The loss forces $M_1$ to be a good model by itself.

When we sample granularity $g$, $M_g$ trains. But the prefix weights are shared – literally the same weights. So what can the suffix weights learn? They cannot contradict what $M_1$ does. They can only add to what the prefix computes.

Concrete example: say $M_1$ learns to predict "cat" with 70% confidence on some image. When $M_g$ runs on the same image, the prefix neurons compute the same thing (identical weights). The suffix neurons can push that 70% up to 85%, or refine the prediction in other ways.

The suffix learns: "given what the prefix already figured out, what correction improves the output?"

In standard multi-task learning, Task A might want weight $w$ to increase while Task B wants it to decrease. They fight. Here, the "tasks" are nested – $M_g$'s task is strictly more expressive than $M_1$'s task. There is no dimension along which they must disagree.

## Distributing widths instead of data

In the original MatFormer paper, you sample granularities randomly per training step – one GPU, alternating widths. Psyche inverts this for distributed training: instead of one machine sampling different widths, *different machines* train at different widths simultaneously.

Imagine a training run with three participants. One has an A100 (80GB), one has a 3090 (24GB), one has an M1 MacBook (16GB). Instead of excluding the weaker hardware, each participant trains at a width that fits their memory.

The A100 runs tier-0 (full width). The 3090 runs tier-1 (half width). The MacBook runs tier-2 (quarter width). All three contribute to the shared prefix; only the A100 trains the suffix.

No big deal, right? Let us expand this idea.

What if, instead of three participants, this expanded to *hundreds*, each with their own hardware constraints, each training at an appropriate tier?

Assuming the gradients from each tier are separable (and they are – each client's forward pass is independent), we can aggregate contributions from all participants and update the model accordingly.

> A heterogeneous training run is equivalent to a MatFormer training run where the tier distribution is determined by hardware rather than random sampling.

## Gradient isolation (what happens to neurons you can't see)

A tier-1 client trains with half the FFN width. What happens to the suffix weights – the neurons it cannot see?

Nothing. Literally nothing.

When training at tier $t$, the suffix weights $W_{\text{suffix}} = W_{\text{up}}^{[m_t:, :]}$ are not in the computational graph. No forward pass touches them; no gradient flows to them.

The chain rule makes this explicit:

$$\frac{\partial L}{\partial W_{\text{suffix}}} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_{\text{suffix}}} = \frac{\partial L}{\partial y} \cdot 0 = 0$$

That is the gradient isolation property – tier-1 clients produce exactly zero gradient for suffix weights. Not approximately zero. Exactly.

When aggregating across tiers, the prefix receives gradients from all clients; the suffix receives gradients only from tier-0 clients. Each region is averaged over its actual contributors.

## Sign-SGD and the voting dynamics

Psyche uses sign-SGD for gradient aggregation. Instead of summing gradients directly, we take:

$$\theta \leftarrow \theta - \eta \cdot \text{sign}\Bigl(\sum_{i=1}^N g_i\Bigr)$$

Each client contributes directional information only. A gradient of $+10^{-6}$ and a gradient of $+10^{6}$ both become $+1$ after the sign operation. This eliminates magnitude imbalance between tiers.

But sign-SGD introduces a subtlety: it turns gradient aggregation into a *majority vote*.

Consider a configuration with 1 tier-0 client and 3 tier-1 clients. For the prefix neurons:

- Tier-0 computes $\nabla L_0$ – the gradient assuming the suffix exists
- Tier-1 computes $\nabla L_1$ – the gradient assuming the model must work alone

These gradients may point in *different directions*. Tier-0's gradient accounts for "how should the prefix change, given the suffix will help?" Tier-1's gradient says "the prefix must do all the work."

With sign-SGD, this becomes a vote: 1 vote for tier-0's direction, 3 votes for tier-1's direction. Tier-1 wins.

| Config | Tier-0 votes | Tier-1 votes | Who dominates? |
|--------|-------------|-------------|----------------|
| (0,0,0,0) | 4 | 0 | Tier-0 |
| (0,0,0,1) | 3 | 1 | Tier-0 |
| (0,0,1,1) | 2 | 2 | Tie |
| (0,1,1,1) | 1 | 3 | **Tier-1** |

This is the "minority tax" – the tier with fewer clients gets outvoted on prefix direction.

Note that the suffix is fine. Only tier-0 contributes to suffix gradients; there is no vote. The problem is prefix-specific.

**When should you care about this?**

If your goal is to maximize tier-0 (full model) quality, you want tier-0 majority. The (0,0,0,1) configuration is often the sweet spot – tier-0 dominates the vote while tier-1 adds sample diversity.

If your goal is compute efficiency (maximize throughput with cheap hardware), you accept some tier-0 quality loss in exchange for running more tier-1 clients.

The good news: when tier-0 and tier-1 gradients *agree* on direction (which they often do, especially as training progresses), both benefit equally. The minority tax only applies when they disagree.

## Fitting this to data

Does any of this actually work? Here is the data.

We ran experiments on a 20M parameter NanoGPT model trained on TinyShakespeare. Five minutes per run on M1 Macs – quick enough to iterate.

The baseline: four tier-0 clients, all full width. For the mixed configurations:

| Config | Tier-0 Lowest Loss | Tier-1 Lowest Loss |
|--------|-------------------|-------------------|
| (0,0,0,0) | 3.764 | — |
| (0,0,0,1) | **3.758** | 3.837 |
| (0,0,1,1) | 3.779 | 3.863 |
| (0,1,1,1) | 3.861 | **3.787** |

Note here that (0,0,0,1) actually *improves* tier-0 loss compared to the all-tier-0 baseline. The single tier-1 client adds sample diversity without corrupting the gradient direction (3:1 vote means tier-0 wins all ties).

At (0,1,1,1), the tier-0 model pays a minority tax of ~0.10 compared to baseline. Tier-1 achieves its best performance because it dominates the prefix vote.

The sweet spot depends on your priorities. If tier-0 quality is paramount, maintain tier-0 majority. If compute efficiency matters more, accept the tax.

## Memory and bandwidth

We do not expect dramatic savings for small models, since embeddings dominate the parameter count. At 20M parameters, embeddings are 67% of the model; tier-1 saves only ~9%.

At production scale, the picture changes:

| Model Size | FFN % | Tier-1 Savings | Tier-2 Savings |
|------------|-------|----------------|----------------|
| 1B | 55% | ~28% | ~41% |
| 7B | 65% | ~32% | ~48% |
| 70B | 70% | ~35% | ~52% |

Note the pattern: savings scale with FFN percentage. At 7B and above, tier-1 saves ~32% memory. That is the difference between "fits on a 3090" and "does not."

Bandwidth follows similarly. After DisTrO compression:

```
Tier-0: 351 KB per FFN layer
Tier-1: 176 KB per FFN layer
```

For a 7B model with 32 clients (8 tier-0, 24 tier-1), total bandwidth drops by 37% versus homogeneous tier-0.

## The schema hash problem

A tier-1 client loading a pre-sliced checkpoint has `intermediate_size = 5504`. A tier-0 client loading the universal checkpoint has `intermediate_size = 11008`.

These are the same model logically, but the configs differ. A naive hash would reject one of them.

The solution: canonicalize before hashing. Restore `intermediate_size` to its base value, force `matformer_tier = 0`. Both clients hash to the same canonical config.

And voila! Compatible clients, despite different checkpoint formats.

## Running heterogeneous training

```bash
cargo run --release -p psyche-centralized-local-testnet -- start \
  --num-clients 3 \
  --config-path ./config/nanogpt-20m-run \
  --client-matformer-tiers 0,1,1 \
  --client-matformer-helper-fractions 0,0,0
```

The tier assignment `0,1,1` gives the first client tier-0 (full width), the others tier-1 (half width).

For implementation details – weight slicing, gradient aggregation, schema canonicalization – see the [Technical Reference](./heterogeneous-training-reference.md).

## Anchoring in prior work

MatFormer has a lineage:

| Work | What it provides |
|------|------------------|
| **Nested Dropout** (Rippel et al., 2014) | "Ordering by truncation" – recovers PCA structure by random truncation |
| **Matryoshka Representation Learning** (Kusupati et al., 2022) | Trains embedding prefixes as strong representations |
| **Universally Slimmable Networks** (Yu et al., 2019) | One set of weights, many widths; the "sandwich rule" |
| **Once-for-All** (Cai et al., 2020) | Train once, specialize many subnets |
| **DynaBERT** (Hou et al., 2020) | Elastic width/depth for transformers |
| **MatFormer** (Devvrit et al., 2023) | Nested transformer FFN; elastic inference |
| **Flextron** (NVIDIA, 2024) | Extends MatFormer with elastic MHA + input-adaptive routing |
| **LlamaFlex** (ICLR 2025) | Zero-shot pruned model generation after brief elastic pretraining |

The US-Nets "sandwich rule" is worth noting: always include the largest and smallest widths each training step. This ensures the extremes are well-trained. Psyche's tier distribution implicitly follows this – as long as you have at least one tier-0 and one tier-N client, both extremes receive gradients.

## The thirty-second version

The small model is not discovered after training; it is trained during training. Each step you randomly choose a width and train that sliced network. So the prefix neurons get trained constantly and are explicitly required to solve the task without the suffix. That breaks permutation symmetry and forces an ordering – very similar to nested dropout and Matryoshka representation learning. The larger widths just add extra neurons that learn refinements, so the objectives are not fighting as much as you would think.

Simple.

## References

1. Devvrit et al. (2023). *MatFormer: Nested Transformer for Elastic Inference*. [arXiv:2310.07707](https://arxiv.org/abs/2310.07707)

2. Kusupati et al. (2022). *Matryoshka Representation Learning*. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)

3. Rippel et al. (2014). *Learning Ordered Representations with Nested Dropout*. [ICML](https://proceedings.mlr.press/v32/rippel14.html)

4. Bernstein et al. (2018). *signSGD: Compressed Optimisation for Non-Convex Problems*. [arXiv:1802.04434](https://arxiv.org/abs/1802.04434)

5. Yu et al. (2019). *Universally Slimmable Networks*. [ICCV](https://arxiv.org/abs/1903.05134)

6. Cai et al. (2020). *Once-for-All: Train One Network and Specialize it for Efficient Deployment*. [ICLR](https://arxiv.org/abs/1908.09791)

7. NVIDIA (2024). *Flextron: Many-in-One Flexible Large Language Model*. [arXiv:2406.10260](https://arxiv.org/abs/2406.10260)
