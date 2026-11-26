---
title: "Robot Learning Notes: from LLMs to Real-World Deployment"
description: "What I learned at CoRL 2026: how robot learning goes from a base model to a robot that keeps learning on the job, plus my own reflection on the week."
date: 2026-11-25
portfolios: [robotics]
---

These notes follow robot learning from the base model up to a robot that keeps learning on the job. Sources: CoRL 2026 talks and posters (Austin, Nov 9 to 12), a Post-CoRL panel in San Francisco (Nov 13), two reflections I wrote, and questions I worked through afterwards. Names or numbers I haven't checked are marked "to confirm".

## 0. The big picture: LLMs vs. robot models

**An LLM gives a robot common sense and language; a robot foundation model has to add perception-to-motor control and learning from physical experience.** The bridge between them is the vision-language-action model (VLA).

### What each kind of model does

|  | LLM / VLM (e.g. Claude Fable, GPT-6 Astra, Gemini) | Robot foundation model (e.g. π0, GR00T) |
| --- | --- | --- |
| Input | Text, images, video | Camera images, robot joint state, an instruction |
| Output | Text or code | Continuous motor commands (action chunks at ~50 Hz) |
| Trained on | Web-scale text and images | Robot trajectories (teleop, gloves, human video) plus a VLM backbone |
| Speed | Seconds per answer | Tens of milliseconds per step |
| Knows | What a mug is, what "tidy up" means | How to reach, grasp and place on a specific body |

### Why a text model can't just drive a robot

- **Physics isn't on the internet.** Force, friction, contact and slip aren't in text or images.
- **Every body is different.** Joints, grippers and cameras differ, so actions don't transfer for free.
- **The world doesn't wait.** A model that thinks for 4 seconds per step fails when the scene keeps moving (Chapter 4: Hang Zhao's "real clock" result; Chapter 5: PEAR at about 4.3 s per step).
- **Demos rarely show recovery.** Robots must learn what to do after a mistake.

### The layered answer: System 2 / 1 / 0

The field's emerging picture splits the robot brain by how fast each layer must react (Hang Zhao, Tsinghua / Galaxea):

- **System 2 (seconds):** a frontier model plans, reasons and talks. This is where LLMs plug in.
- **System 1 (tens of Hz):** a VLA or world action model turns the plan into motion.
- **System 0 (hundreds of Hz):** whole-body control, balance, contact and force.

My one-line version: *LLMs gave robots a brain; the race now is the body and real-world experience.*

### Where does sim data fit? (my question)

LLMs are now also trained on synthetic data, but robots need simulation for something different: practicing actions, not reading about them. Simulation is cheap and safe, but its physics never quite match reality (the sim-to-real gap), so real-robot data and real-robot testing stay the ground truth (Chapters 3 and 7).

## 1. Foundation: backbones and pretraining

**Robot models inherit their "brain" from vision-language models, then learn to act.** Each jump in LLMs/VLMs makes robot models smarter at understanding, almost for free.

### 1.1 From VLM to VLA

- A VLA starts from a pretrained vision-language model and adds an **action expert** that outputs motor commands. π0 builds on PaliGemma and outputs continuous action chunks.
- What carries over: object names, spatial relations, following new instructions. RT-2 showed a robot following instructions it never saw in robot data.
- What doesn't: contact, force and timing. Those have to come from robot data (Chapter 2).

### 1.2 How actions are generated

- **Diffusion / flow matching:** start from noise and refine it into a smooth chunk of motion. Good for multi-modal behavior (several valid ways to do a task).
- **Autoregressive (AR):** output motion token by token, like text. Easier to merge with LLM training. At CoRL 2026, my read was that **AR scored a point** in the AR vs. diffusion debate.
- **Action tokenizers** turn motion into tokens. Physical Intelligence's FAST compresses action chunks. Galaxea's **ActionCodec2** (Hang Zhao's group) streams tokens so the robot reacts mid-motion: in their demo a chunk tokenizer missed a moving block by 7.4 cm, the streaming one by 0.0 cm.

### 1.3 Co-training: don't forget the web

- Fine-tuning only on robot data erases web knowledge (catastrophic forgetting).
- π0.5 trains on a mix of robot data, web vision-language data and verbal instructions, so it keeps its understanding while learning to act.

### 1.4 Reasoning on demand

- Chain-of-thought (writing out reasoning before acting) helps hard tasks but costs time.
- Galaxea **G0.5**: on about 90% of tasks, skipping reasoning gives the same success at 2x speed. A 5-stage air-fryer task went from 1/5 to 3/5 with reasoning. So reason only when the task needs it.

### 1.5 The model landscape (as of Nov 2026)

**Announced, downloadable and usable are three different things.** Most frontier robot models are announced with demos, offered to partners or a waitlist, and not released as weights.

| Company | Model | What it is | Access |
| --- | --- | --- | --- |
| Physical Intelligence | [π0.7](https://arxiv.org/abs/2604.15483) (Apr 2026) | Generalist VLA that combines learned skills and follows guidance | Paper; PI also runs an API beta |
| Google DeepMind | [Gemini Robotics 2.5, ER 2, On-Device 2](https://deepmind.google/models/gemini-robotics/) | Split into robot action, embodied reasoning and on-robot execution | 2.5 to partners; ER 2 preview in AI Studio; On-Device 2 for trusted testers |
| NVIDIA | [GR00T 2](https://research.nvidia.com/labs/gear/gr00t-2) | Predicts future video and robot actions together | Code due December 2026 |
| NVIDIA | GR00T N1.7 | Earlier open humanoid model | Downloadable; Apache-2.0 code, NVIDIA Open Model License weights |
| Generalist AI | [GEN-1.5](https://generalistai.com/blog/gen-1.5) (Aug 2026) | Learns new tasks from small demo sets | Waitlist |
| Dyna Robotics | [Dyna-2 / Dyna-2.1](https://www.dyna.co/dyna-2.1) | Dyna-2 is a world action model; Dyna-2.1 is the agent built on it | Commercial |
| Genesis AI | [GENE-26.5](https://www.therobotreport.com/genesis-ai-introduces-gene-foundation-model-more-dexterous-manipulation/) (May 2026) | Trained on glove hand data, egocentric and internet video, and sim | Not released as weights |
| Skild AI | [Skild Brain](https://www.skild.ai/) | "Omni-bodied": one brain for many robot bodies | Commercial |
| Figure | [Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) (Sep 2026) | Co-designed with Figure's humanoid | Runs on Figure robots |
| Galaxea AI | [G0.5](https://github.com/OpenGalaxea/GalaxeaVLA) (Jun 2026) | Reasoning and action tokens in one stream (see 1.4) | Weights out, non-commercial licence |

Tesla's Optimus is a humanoid program, not a standalone model you can compare.

### 1.6 Open starting points

**These play the role BERT and GPT-2 played for language: common bases anyone can build on.** Check the model licence separately from the code licence; they often differ.

| Starting point | What it is | Licence |
| --- | --- | --- |
| [openpi](https://github.com/Physical-Intelligence/openpi) | Training and inference code with π0, π0-FAST and π0.5 checkpoints | Apache-2.0 |
| [AIRoA-0](https://huggingface.co/airoa-org/AIRoA-0) (Nov 2026) | 3.36B VLA trained on handheld-gripper demos | Code Apache-2.0; weights under Gemma terms; dataset listed as coming soon |
| [SmolVLA / LeRobot](https://huggingface.co/lerobot/smolvla_base) | ~450M VLA plus Hugging Face's robot learning framework | Apache-2.0 |
| [MolmoAct2](https://github.com/allenai/molmoact2) | AI2's robot action model with training and inference code | Apache-2.0 |
| [OpenVLA](https://github.com/openvla/openvla) | Pretrained VLA | Code MIT; models under the Llama licence |
| [Octo](https://github.com/octo-models/octo) | Generalist policy with pretrained checkpoints | MIT |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | A method for generating actions with diffusion, not a pretrained brain | MIT |
| [ACT / ALOHA](https://github.com/tonyzhaozh/act) | Action chunking for imitation learning, plus a two-arm data collection setup | MIT |

### 1.7 How actions are represented

**What a model outputs decides how far it travels between robots.**

- **Joint commands vs. gripper poses.** Joint angles are specific to one arm. A target pose for the gripper (where it should be and how it should face) carries over to other arms better, but each robot still needs its own layer to turn that pose into motor commands it can actually reach.
- **Action chunks.** Predicting several future actions at once means fewer calls to the big model. The open design question is how much of a chunk to run before looking again: run too much and the robot misses changes; too little and it is slow. Streaming tokenizers like ActionCodec2 (1.2) are one answer.
- **Latent actions.** A model can learn compact codes for "what changed between two frames" from plain video, with no motor labels. That unlocks human and internet video as training data, but mapping those codes onto a real robot body is still separate work.

## 2. Data: the fuel

**The bottleneck is shifting from how much robot data you have to what kind: recovery, contact-rich and real deployment data.**

### 2.1 The data pyramid, today and tomorrow

Russ Tedrake (MIT, Walden Robotics) framed it as a pyramid that flips:

| Layer | Today | In the limit |
| --- | --- | --- |
| Language and vision (internet) | Huge base | Bottom layer |
| Ego video, sim, UMI, gloves, general robot data | Middle | Middle |
| Teleoperation | Small tip | Middle |
| Deployment data (the actual job) | Almost none | **Largest layer** |

"In the limit, deployed robot data will dwarf everything else," because training on the job makes the training distribution match the test distribution.

### 2.2 Quality before scale

Boston Dynamics' closing message: robot behavior is held back more by data **quality** than amount. High quality means:

- **Agile, dynamic moves**, not only slow careful ones.
- **Recoveries and "unhappy path" demos**: what to do when the object slips or the robot gets bumped.

Where they want investment: better teleoperation (low lag, full control, trained operators), better UMI-style handheld tools (dexterous, high-force, for humanoids), and a recipe for RL-generated synthetic data.

Related: **Behavior Uncloning** (Bangya Liu, UW-Madison / Abaka.ai) removes unwanted behaviors that imitation copies from demos, baked into the weights.

### 2.3 Sources of data, ranked by value

Implied by Tedrake's talk: field interventions and corrections > field teleop > lab teleop > human video / sim.

- **First-person teleop:** the operator sees through the robot's cameras (often VR), so data matches the robot exactly. Highest quality, highest cost.
- **Exoskeletons:** UME (James Xu, Stanford) is a low-cost arm exoskeleton with real-time force feedback; it retargets to OpenArm, Franka and xArm.
- **Human / egocentric video:** cheap and huge, but has no robot actions; it must be mapped onto robot hands. NVIDIA's open **Human-1M** dataset (scale unit to confirm), EgoHTR for legged walking data (Jonas Frey).
- **Touch data:** FingerEye fingertip sensors output contact-depth maps: 65.9% vs. 24.1% average success in sim against wrist camera + regular touch.

**Data is already a supply chain.** Maxinsights, which pays people to record everyday work with cameras, [says](https://theaiinsider.tech/2026/10/23/maxinsights-raises-30m-seed-round-and-launches-physical-ai-data-tools/) all 200,000 hours of egocentric data behind Genesis AI's GENE-26.5 came from it, and that it is the largest data provider for Dyna-2. Both are the supplier's own claims.

### 2.4 Mid-training data is commoditizing

Tedrake's honest warning: as base models get stronger ("Astra, Fable, ... oh my"), they need less generic teleop/ego data, and open datasets like Human-1M lower its price. What stays valuable: on-the-job deployment data, cross-embodiment variety, dexterous/contact data and recovery data.

## 3. World models and simulation

**World models let robots imagine the next moment, which helps training, planning and evaluation, but they are not yet reliable enough to replace the real world.** My CoRL read: world models are still short on practical use, and world action models (WAMs) are still a hyperparameter-tuning game.

### 3.1 What a world model has to do (Fei-Fei Li, Stanford / World Labs)

1. **Represent** the scene: objects, 3D layout, relations.
2. **Predict** the consequences of an action.
3. **Simulate** the next state, so robots can practice and be tested inside the model.

Three uses: rendering, planning (try actions in imagination) and simulation (a stand-in world for training and testing).

### 3.2 Reconstruction and generation are one problem

- World Labs' new model (name "Atlas" to confirm; not Boston Dynamics' humanoid) does both with one architecture: dense reconstruction (many photos), sparse reconstruction (a few photos, it fills the rest), pure generation (text only).
- Slide line: "Viewpoint is all you need." If a model can render any viewpoint correctly, it effectively knows the 3D world.
- Architecture: a multimodal autoregressive diffusion transformer. It generates one view at a time, each checked against earlier ones, to fix **spatial consistency** (objects shifting between angles).
- Why it matters: a few phone photos of a real lab or factory cell could become a usable 3D copy (real-to-sim).

### 3.3 Imagination at training time, speed at run time

- **Fast-WAM** (Tsinghua MARS Lab, NeurIPS 2026): learn video prediction and actions together, then skip video at run time. 190 ms per step, about π0.5's 180 ms, vs. 580 to 810 ms for WAMs that still generate video.
- **Large Video Planner** (MIT / Berkeley / Harvard, Yilun Du and others): a 14B video model generates a human hand doing the task, then the hand motion is mapped to the robot. Open LVP-1M dataset (1.4M clips). Limits: slow, open-loop, hand-to-robot mapping errors.
- **1X world model** (Chaoqi Liu): a video model that predicts what the NEO humanoid will see next, used to evaluate policies without running the real robot, and in 2026 as a policy itself.

### 3.4 The catch: sim-to-real

If the copy is slightly off, a policy that passes in sim can still fail on the real robot. Academics told me there's often little to do besides sim, because real hardware is scarce. That's why every world-model or sim result still needs a real-robot check (Chapter 7).

## 4. Policy architecture: how the robot brain is built

**Split the robot brain by clock, not by function, and keep the fast layers fast.** Robots mostly fail at the hand-offs between layers.

### 4.1 Five ways to use a frontier model (Hang Zhao)

The same frontier model can be used more or less often during a task (RoboCasa365 kitchen tasks, about 59 s each):

1. **Coding policy** (0 calls): writes a program once, then the program runs.
2. **Coding + ask** (~1 call): asks the model only when stuck.
3. **Orchestrated** (~4 calls): the model picks the next skill; pre-built skills do the motion.
4. **Direct** (~18 calls): the model says where the hand goes next, about every 2 s.
5. **Agentic** (~18 calls): looks, computes, moves, like a coding agent in a loop.

With the clock running, policies that call the model every step collapse. Orchestrated was best either way.

| Policy mode | Success, clock paused | Success, clock running |
| --- | --- | --- |
| Coding | 45% | 45% |
| Coding + ask | 65% | 45% |
| Orchestrated | 75% | 65% |
| Direct | 60% | 15% |
| Agentic | 60% | 20% |

### 4.2 Split by clock

| Layer | Reacts in | What it handles | Example models |
| --- | --- | --- | --- |
| System 2 | Seconds to minutes | Task state, language, scene | GPT-6 Astra, Gemini Robotics-ER, coding agents |
| System 1 | Tens of Hz | Hand-object relative motion | π0.5, Galaxea G0.5, world action models |
| System 0 | Hundreds of Hz | Contact and force | SONIC, BeyondMimic, whole-body trackers |

- "A slipping cup is perception and control at once, but it changes in milliseconds: System 0."
- The idea echoes the 3T three-layer architecture (E. Gat, 1998); what's new is that every layer is now learned.
- Different jobs need different mixes: an assembly line is mostly System 0; housework needs a lot of System 2.
- His ask: layers should pass **intent down and feedback up**, and each should close its own loop. Today almost nothing flows back up.

**My insight: System 2 in the cloud, Systems 1 and 0 on the edge.** System 2 only answers every few seconds, so a 50 to 200 ms network round trip doesn't matter. The fast layers can't wait for the internet and must run on the robot (Jetson-class compute). If the connection drops, the robot still moves safely. The cloud-to-edge hand-off is the slowest seam, so it needs the most testing.

Two real reference points: Boston Dynamics' Atlas behavior model controls the full robot [at 30 Hz](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/) (System 1), and the Franka FR3 Duo exposes [1 kHz low-level control](https://franka.de/fr3-duo) (System 0).

### 4.3 One whole-body model, not a skill library (Boston Dynamics)

- Hard lines between skills are a fallacy. "Grasp an object" can't be separated from reaching (high or low), walking (moving object), active perception (hidden object), search (clutter) or balance (heavy object).
- So: "put everything through the one-model woodchipper", then solve two new problems: **steerability** (telling one big model exactly what to do now) and **compositionality** (combining skills it learned separately).

### 4.4 Memory and long-horizon tasks (the biggest CoRL trend)

- **MEM** (Physical Intelligence + Stanford, on π0.6): short-term compressed video plus a long-term text summary. Tasks up to 15 minutes, under 300 ms per step. One hour of 4 cameras would be 184M tokens raw, so memory must be compressed.
- **MessyMem** (Stanford, Anuva Banwasi): 3D scene graph plus keyframes; 3+ hour missions, ~10k keyframes, 80.0 vs. 65.2 next best.
- **MemoryVAM** (UCLA): episode memory for video action models; real counting tasks went from 0% to 95 to 100%.
- **Lifelong VLA** (SCUT + Trento): short- and long-term LoRA adapters with a learned gate; 83.2% success, 11.4% forgetting on LIBERO.
- **Map-conditioned VLAs**: give the model a map of the room alongside camera images for room-scale mobile manipulation (compared against π0.5).

### 4.5 Small, fast VLAs on the robot

- π0.5 takes about 1.4 s per step on a Jetson Orin (0.7 Hz), too slow for System 1.
- **EagleVLA** (Peking U.): predicts where the world will be when actions run and skips the big model when confident; 14.7x faster on Jetson Orin.
- **BitVLA** (CAS ICT): 1-bit weights (-1, 0, +1); 11x less memory, 4.4x faster than OpenVLA-OFT.
- Other small models: SmolVLA (450M), Octo (27M to 93M). Ways to shrink: distill, quantize, or skip slow parts at run time as Fast-WAM does.

## 5. Post-training: making the model better

**Robot learning is copying the LLM playbook: pretrain, fine-tune, then improve with reinforcement learning from experience.** The open question is what RL actually teaches.

### 5.1 Are LLMs moving into RL? (my question)

Yes, in two ways. LLMs are being post-trained with RL to act as robot policies (PEAR below). And LLMs help RL itself by writing rewards, designing curricula and judging success from camera images.

### 5.2 RL in simulation: PEAR (MIT, Song Han's lab)

Can a general open reasoning model learn manipulation from sim RL alone, with no human demos?

- **Setup:** Qwen3.8-27B sees the instruction, camera image and arm state, reasons step by step, and outputs JSON gripper moves (cm, rotation, open/close). Inverse kinematics and a low-level controller execute them. No detector, grasp planner or skill library.
- **Training:** GRPO with 8 rollouts per task, 128 groups per update, reward 1 for success and 0 for failure; groups that are all-success or all-fail are resampled. Curriculum: LIBERO basics, then long tasks, then RoboDojo grasping and block alignment.
- **Results:** LIBERO success 1.6% to 74.4%, close to GPT-6.1 Sol (76.2%) and GPT-6 Astra (80.6%) at max reasoning. 18/30 on a real dual-arm robot. On a new task in a familiar scene, PEAR 57.4% vs. 0.75% for demo-fine-tuned π0.5, though π0.5 is still stronger on original LIBERO.
- **Bottleneck:** about 3,230 thinking tokens, about 4.3 s per step on GB300, vs. 50 Hz VLA control. Lower reasoning effort drops success from 74.4% to 21.9%.
- **My open question:** what did RL teach it: better 3D localization, action scale, progress tracking, or error recovery? Where does long reasoning help, and why does shortening it fail?
- **Read:** LLMs enter robotics through RL post-training, but at seconds per step they work as System 2 planners while fast VLAs keep control. Names and numbers to confirm against the paper.

### 5.3 RL from real runs

- **CLIFT** (Yuxin Chen, UC Berkeley): adapts Gemini Robotics On-Device into a humanoid specialist without touching its weights. Reward feedback from real runs becomes new training data, loop after loop. One of the first studies of adapting a closed, API-only robot model on a real humanoid.
- π*0.6 with Recap (Physical Intelligence) is the industry version: the robot works, humans correct failures, a value function scores outcomes, and the policy improves on its own experience.

### 5.4 LLMs as coaches

- Foundation models reason about what went wrong, write reward code and judge progress from images. Negar Mehr's lab (UC Berkeley, to confirm) showed an LLM designing a curriculum to teach a humanoid to run backward: stability, walk, speed up, target task, with reward code for each step, trained in sim and run on a real small humanoid.
- Caveat: VLM judges are still unreliable on fine physical detail like grasp stability.

### 5.5 Reusing what the robot already knows (RobIn lab, UT Austin, to confirm)

- **Reuse is a trade-off.** Learning to hammer from an earlier "banging" skill: freezing the skill learns fastest but caps success around 80%; fine-tuning it starts slower but reaches about 100%; from scratch is slowest.
- Ask "how much more can it learn from here, and how fast", not only "how good is it now".
- **Modes as an action space:** index the distinct ways a diffusion policy can do a task and explore among them (IMP-Steer, STRATA).
- **SafeMimic:** learn the steps from human video, then practice safely on the robot's own body, backtracking when an action looks unsafe.
- **OopsieRecover:** a short spoken correction ("avoid hitting the lid") becomes a small program that steers the VLA without retraining; success after 5 interventions.
- **Continual learning with LoRA:** keep the big model frozen and save a small adapter per task as its memory, so old skills aren't forgotten.

### 5.6 Multiple robots learning together (Negar Mehr's lab, to confirm)

- Behavior cloning works worse with two agents: one agent's small errors change what the other sees, and errors compound.
- Coordination is multi-modal: if two arms each copy a different valid route, the hand-off fails.
- Fix: freeze each robot's diffusion policy and learn only a small **residual** correction for coordination.
- Continual learning as consensus: merge old and new weights by how much each matters for its task, with a **moving anchor** that keeps the robot tied to everything learned so far.

## 6. Deployment: learning on the job

**The science works; the hard part now is a product that meets the customer's reliability bar and keeps improving in the field.** Most of this chapter comes from Russ Tedrake's keynote, "Learning in the factories".

### 6.1 The customer sets the bar, over a full shift

- 98.5% uptime (about 7 minutes of downtime in an 8-hour shift).
- Under 0.5% defects (fewer than 1 in 200 tasks wrong).
- Zero safety incidents, and no slowing the line.
- Research papers report "8/10 successes"; a factory needs 199/200, all shift, at line speed. That gap is the whole product problem.

### 6.2 Which tasks to pick

Boring and labor-intensive, but with expensive failures. High volume gives clear value and many repetitions to learn from; expensive failures mean customers will pay for proven reliability. The product is reliability, not just capability.

### 6.3 Human-in-the-loop: HG-DAgger

- **DAgger** (Ross et al., 2011): let the robot run, have an expert label what it should have done in the situations it actually got into, retrain, repeat. It teaches recovery from the robot's own mistakes.
- **HG-DAgger** (human-gated, Kelly et al., 2019): a human takes over only when the robot is about to fail, then hands back. Only the takeover segments are added to training.
- Walden ships a new model **every Thursday**: work all week, log interventions, retrain, release, measure again.
- Metric: **mean time between interventions**, from about 5 s in week 1 to about 80 s in week 6 (roughly 15x on a log scale). A full shift is 28,800 s, so there is still a long way to go.

**The research line behind learning on the job:**

| Year | Work | Idea |
| --- | --- | --- |
| 2011 | DAgger | Get expert labels for the states the learner actually visits |
| 2019 | HG-DAgger | The human decides when to take over and corrects |
| 2023 | [Sirius](https://arxiv.org/abs/2211.08416) (RSS) | Use interventions during deployment to keep improving the policy |
| 2024–25 | [HIL-SERL](https://arxiv.org/abs/2410.21845) | Real-world RL plus human demos and corrections |
| 2025 | [RECAP / π*0.6](https://arxiv.org/abs/2511.14759) (Physical Intelligence) | Improve a VLA from demos, corrections and its own experience |
| 2026 | [Learning While Deploying](https://arxiv.org/abs/2605.00416) | Fleet-scale RL: a whole fleet's experience and interventions improve one generalist policy |

The new step is sharing what one deployment learns across tasks and sites, instead of fixing each robot alone.

### 6.4 Stronger pretraining learns more on the job

| Pretraining | Before adaptation | After task adaptation | After 24 h on the job | Remote intervention time | Autonomous after 24 h |
| --- | --- | --- | --- | --- | --- |
| 1k hours | ~10 s | ~27 s | ~28 s | 2.22 h | 67% |
| 10k hours | ~8 s | ~40 s | ~56 s | 1.78 h | 80% |
| 100k hours | ~11 s | ~41 s | ~80 s | 1.36 h | ~89% |

Mean time between interventions, from [Walden's LBM 2.0 post](https://www.waldenrobotics.com/news/lbm-2-0-accelerating-learning-in-deployment) (Nov 11, 2026): three 4.7B models, about 6.2 hours of demos per task, seven tasks, three 8-hour shifts each.

- The gap opens **during deployment**, not before: 10k and 100k look the same after adaptation.
- Stronger models get less intervention data but improve more, and their interventions are different: short nudges on rare edge cases instead of long takeovers. So also track intervention duration and % of time autonomous, not only time between interventions.
- Training happens in stages: pretraining (internet), mid-training (teleop + ego data), then deployment learning.

### 6.5 Will the flywheel stall?

The honest counter-argument: deployment needs 97 to 99.9% success, so companies narrow the task, the narrow task saturates, and new data stops teaching anything new. The answer from the results above: a strong pretrained base keeps learning instead of saturating.

### 6.6 From 1:1 to few:many

Today one human watches one robot. To scale, the policy detects when it's unsure and **asks for help**; a few remote operators answer the robots that call. That needs a reliable "I'm not sure" signal, a remote ops setup, and robots that pause safely.

### 6.7 The next bottleneck is building robots

Walden's AI is no longer the main limit; manufacturing is. Their robots help assemble robots, they design for mass production, and they partner with Toyota.

### 6.8 Safety and security

**A robot's mistakes are physical, so safety can't live only inside the model.**

- **Physical limits.** Caps on force, on the space the robot can reach and on how close it gets to people, plus clear stop conditions. The fastest checks belong in System 0, the controller itself; slower ones belong in uncertainty monitoring and asking for help (6.6).
- **Misleading inputs.** A doctored image, sticker or reflection can change what a vision model sees and so what it decides. Robot models inherit the adversarial problems of vision models.
- **Instructions vs. text in view.** A robot that takes language instructions must tell an authorized command from words it merely sees, such as a sign or a label saying "open the door". This is prompt injection with a body.
- **Proposing vs. permission.** A model suggesting an action and the system allowing it to run should be separate decisions, with the permission check outside the model.

## 7. Evaluation: measuring what's real

**You can benchmark an LLM offline; you can't benchmark a robot policy that way.** Every talk above ended up needing a real-robot check.

### 7.1 Run the real clock

Hang Zhao's one request to researchers: in simulation, report success before a deadline with thinking time counted. Stop pausing the clock. A real robot can't pause.

### 7.2 What good robot evaluation measures

| Measure | Why it matters | From |
| --- | --- | --- |
| Success with the real clock running | Slow policies collapse in real time | Hang Zhao |
| Uptime, defect rate, safety over a full shift | The customer's actual bar | Russ Tedrake |
| Time between interventions, intervention length, % autonomous | How much human help a deployment needs | Russ Tedrake |
| Learning curve over real trials, not only final success | Reuse trades speed for ceiling | RobIn lab |
| Corrections needed before success | Cost of fixing a failure in the field | OopsieRecover |
| Ask-for-help quality: missed failures vs. false alarms | Needed for few:many operation | Russ Tedrake |
| Real vs. sim (or world-model) score on the same task | Whether the copy can be trusted | Fei-Fei Li, 1X |

### 7.3 Task types worth testing

- **Long-horizon tasks with memory:** an object moved out of view, counting, "find where I put X" (MEM, MessyMem).
- **Room-scale mobile manipulation** (map-conditioned VLAs).
- **Contact-heavy and dexterous tasks** (FingerEye, System 0).
- **Blended skills:** grasp while moving, grasp from clutter (Boston Dynamics).
- **Coordination and hand-offs** between two arms (multi-agent learning).
- **Cross-embodiment:** the same model on different robot bodies.

### 7.4 Why robots fail at the seams

Each layer can work alone, yet things break where one passes control to the next: System 2's plan doesn't match what System 1 can do, or System 1's motion doesn't match what System 0 can track. You only find these failures by running the whole stack together, in real time, on real hardware.

### 7.5 Sim tools and benchmarks

| Tool or benchmark | What it is |
| --- | --- |
| [SuperDex](https://github.com/facebookresearch/project_superdex) (Meta) | A unified platform for dexterous manipulation research; VR teleop (Quest 3) is on its roadmap |
| Cosmos (NVIDIA) | An open platform for building world models. Predicting video is not the same as simulating physics |
| [ToyBlocks](https://allenai.org/blog/toyblocks) (AI2, Nov 2026) | Sim framework with 85,336 long-horizon demonstrations, Apache-2.0 |
| [DuoBench](https://duobench.github.io/) | Two-arm benchmark on the Franka FR3 Duo: 11 sim tasks, 4 also run on real robots |
| [RoboColosseum](https://robocolosseum.ai/) | Shared real-robot evaluation across many sites; an open track ranked by pairwise ratings and a fine-tune track (100 demos, 25 attempts) |

Two older routes still frame most of this: **sim-to-real** (train in sim, run on the robot) and **real-to-sim-to-real** (rebuild the real scene in sim, train there, go back).

### 7.6 How to read a success rate

**"90% success" means little on its own.** Before trusting a number, ask:

1. How many attempts? Real-robot trials are expensive, so a handful of runs can swing the result.
2. Were the objects and scenes familiar or new?
3. Who reset the scene between attempts, and how?
4. Did thinking time count, or was the clock paused (7.1)?
5. What counted as success, and who judged it?
6. How long did it take? A slow success can still fail the job.

Example: Generalist reported that [GEN-1](https://generalistai.com/blog/gen-1) lifts average success to 99% on tasks where earlier models reached 64%. That holds for those selected tasks under those conditions, not for manipulation in general.

## 8. Hardware and embodiment

**Embodied AI is a system, not a model: body, hands, sensing and compute all have to work together.** Hardware moves at hardware speed, not software speed.

### 8.1 Three pillars (Wolfram Burgard, UTN)

Burgard co-wrote *Probabilistic Robotics* (with Sebastian Thrun and Dieter Fox). His key technologies for embodied AI:

1. **Humanoid robot:** the body.
2. **Robotic hand:** dexterous manipulation, the hard part of useful work.
3. **Compute for generative AI:** to train and run the models.

His full loop, for "pick up the red mug and put it in the sink": perception (vision, language, **touch, force**) goes into one multimodal model, which plans steps and safe motion, executes, and keeps correcting from sensor feedback. Touch and force are first-class inputs, not extras.

### 8.2 Don't throw away the probabilistic mindset

- A robot never knows the world exactly, so it should track how confident it is. That idea from *Probabilistic Robotics* became a basis for self-driving cars.
- Classic localization is already factory-grade: an 11-ton KUKA omniMove carrying concrete beams knows its position to about 1 mm and 0.01 degrees, in static and busy halls alike.
- His scorecard for the classic self-driving pipeline: localization solved best, then mapping; scene understanding, prediction and planning stayed weak. That gap is what learned models are filling, but they still have to reach this level of reliability.

### 8.3 Hand design shows which era a company bets on (my CoRL read)

- **Boston Dynamics bets on now:** a 3-finger, low-degree-of-freedom industrial hand, maximum strength and stability. Its DNA is System 0.
- **Tesla Optimus bets on 5+ years out:** an exact copy of the human hand for maximum generality. If that works, handheld tools like UMI become unnecessary and human motion retargets directly.
- From what I saw and heard, Chinese dexterous hands have split into tiers. Sharpa seemed to lead on stability at about $50k per hand, and I heard North American big tech wants to buy it in bulk.

### 8.4 Demos reveal maturity

At CoRL, Optimus appeared to me to be teleoperated (blamed on "network issues"), while Gemini ran a real policy rollout, jerky motion and all, and took live challenges from the crowd.

### 8.5 Compute on the robot

System 1 and 0 must run on the robot, so on-board compute (Jetson Orin / Thor class) and small, fast VLAs (Chapter 4.5) decide what a robot can actually do in real time. NVIDIA's open stack includes reference hardware alongside its GR00T 2 model.

### 8.6 Sensing beyond cameras

**Cameras miss slip and force, so the best fix is often a new sense, not a smarter camera model.**

- **Vision plus touch at the fingertip:** [FingerEye](https://arxiv.org/abs/2604.20689) puts binocular cameras behind a soft contact surface, so the robot sees while approaching and feels after contact (2.3).
- **Touch where vision fails:** [DAVATaR](https://davatar-corl26.github.io/) (UT Austin) works across the air-water line, where refraction, reflections and foam fool cameras. It pairs fluid-based touch sensing with training images that mimic those distortions.
- **Touch doesn't transfer yet:** a model trained on one tactile sensor often fails on another, the way actions fail to transfer between arms.
- **Muscle signals:** wristbands that read muscle activity can show intended movement and effort, but need interpretation and calibration per person.
- **Scene representations:** besides images, point clouds and Gaussian splats (many small colored 3D blobs) capture geometry better, at the cost of building and updating them.

### 8.7 Hard objects and body shapes

- **Deformable objects** like cloth, rope and food change shape when touched, so they are harder to track, predict and simulate than rigid ones.
- **Grippers vs. hands:** two-finger grippers are simple; multi-finger hands add joints, contact points and much harder data collection.
- **Body shapes:** fixed arms work at one station; arms on wheels move between stations without legs; humanoids target spaces and tools built for people; drones reach places ground robots can't, under tight weight, power and latency limits.
- **Fleets:** several robots sharing tasks and information need task allocation, delay handling and failure recovery, and some research coordinates them without a central controller (5.6).

## 9. The field: trends and outlook

**Models are converging; the physical layer isn't.** As LLMs get stronger, value moves to real-world data, evaluation, deployment and hardware.

### 9.1 How fast LLM progress changes robot learning (my question)

1. **Better backbones:** each VLM jump makes robots better at understanding scenes and following instructions, with less robot data.
2. **Planners and orchestrators:** LLMs break long tasks into steps, spot failures in images and pick the next skill. A robot skill becomes another tool.
3. **Data engines:** VLMs label and score robot video, write sim environments and reward functions, and pull signal from human video.
4. **The recipe:** transformers, scaling laws, pretrain then fine-tune, RL post-training, all copied from LLMs.
5. **Judges:** VLMs score success at scale, but miss fine physical detail.
6. **Cheaper robot software:** coding agents write drivers and integration code, so supporting many robot types gets easier.

What LLM progress doesn't solve: physical interaction data, 99.9% reliability, the sim-to-real gap, hardware cost curves and real-world evaluation.

### 9.2 Open ecosystems (NVIDIA)

NVIDIA's keynote (speaker to confirm) closed with GR00T 2 plus five open pieces: SimFoundry (simulation), ASPIRE (agent tooling), FLORA (annotation), Human-1M (open human data) and reference hardware. Goal: physical AGI. The play: give away the model, data and tools so everyone builds on NVIDIA chips. When models and data are open, the bottleneck moves to real robots and fair testing.

### 9.3 CoRL 2026 trends (running list)

1. Memory for long-horizon tasks (the biggest).
2. Spatial-map-conditioned VLAs.
3. One whole-body model instead of separate skills.
4. Data quality over data scale.
5. Richer touch sensing.
6. Reusing prior knowledge and safe real-world learning.
7. Small, fast VLAs on the robot.
8. Video and world models as planners.

### 9.4 My CoRL reflection (观感流水账)

My one-line read: CoRL 2026 was the year the field stopped asking whether robot learning works and started asking how to ship it.

**1. Deployment is the new finish line, and customers set the bar.** Russ Tedrake (Walden) said the technology is ready. The hard part is a product that holds 98.5% uptime and under 0.5% defects over a full shift. Boston Dynamics disagreed on one point: data quality, not scale, is still what holds robots back.

**2. Russ's keynote: weak pitch, strong data.** He spent most of an hour on an ad for Walden, and almost forgot he is also an MIT professor. But his HG-DAgger results were the most useful numbers of the week. Models with stronger pretraining kept improving on the job and needed about 40% less operator time.

**3. Live demos show how mature a team really is.** From what I saw, Tesla's Optimus kept being teleoperated, blamed on "network issues", so Tesla still looks image-first. Gemini ran a real policy rollout, jerky motion and all, and took live challenges from the crowd. I trust the jerky one more (see 8.4).

**4. Hand design shows which era a company bets on.** Boston Dynamics bets on now: a three-finger, low-DoF industrial hand built for strength and stability. Optimus bets on five-plus years out: an exact copy of the human hand. From what I saw and heard, Chinese dexterous hands have formed clear tiers, and Sharpa leads on stability even at about $50K a hand (see 8.3).

**5. The clock is the real constraint.** Hang Zhao's "split by clock, not by function" is the idea I keep coming back to. Policies that call a frontier model every step fell from 60% to 15–20% once the clock kept running. PEAR needs about 4.3 s of thinking per step. My read: the big model plans in the cloud, a small fast model moves on the robot, and robots fail at the seams between them.

**6. World models: exciting, still not practical.** Fei-Fei Li made the best case: reconstruction and generation are one problem. But world models still lack real-world use, and world action models are still a game of tuning hyperparameters. In the AR vs. diffusion debate, autoregressive scored a point this year.

**7. Memory was the biggest research trend.** MEM ran 15-minute tasks and MessyMem ran 3-hour missions. Robots that only react to the current camera frame are on their way out.

**8. Academia now chases trends as fast as Bay Area startups.** World models two years ago, real2sim and sim2real last year, agentic AI next. Academic friends say there is little they can do besides sim, because real robots are scarce. That is a little sad.

**9. Robot companies finally found a smarter target.** 「机器人公司终于找准了新目标：机械化程度低、工作时间短的劳动密集型行业。」 Robots are slow but can work long hours. That beats warehouse logistics, which looks easy to automate but is crowded and thin-margin.

If I had to bet after this week: the winners will pair a strong pretrained base with real deployment data, and will test on real robots in real time.

### 9.5 Is robot manipulation becoming a foundation-model skill?

My PEAR write-up asked whether 具身智能 could be "absorbed" by general models like Qwen. The evidence so far: general reasoning models can learn manipulation through sim RL and generalize to new tasks better than demo-trained VLAs, but they are two orders of magnitude too slow for control. The likely near-term shape is layered: a general model as System 2, a fast VLA as System 1, contact control as System 0.

## Cheat sheet

| Concept | One line |
| --- | --- |
| VLA | A vision-language model plus an action expert that outputs motor commands |
| Co-training | Mix web and robot data so the model keeps its common sense |
| Action tokenizer | Turns motion into tokens; streaming tokens let the robot react mid-motion |
| System 2 / 1 / 0 | Plan in seconds, move at tens of Hz, control contact at hundreds of Hz |
| Orchestrated policy | The big model picks skills; fast skills do the moving |
| Whole-body model | One model for all skills, because skills blend into each other |
| Memory | Compressed short-term video plus long-term summaries for tasks of minutes to hours |
| Edge VLA | A small, fast policy that runs on the robot's own computer |
| World model | Predicts the next state of the world for training, planning and testing |
| WAM | A world action model: predicts future video and actions together |
| Sim-to-real gap | Policies that pass in simulation can fail on real hardware |
| Data quality | Recovery, agile and contact-rich demos matter more than more of the same |
| GRPO / sim RL | Reward success, compare rollouts in a group, improve without human demos |
| LoRA adapter | A small add-on per task, so the base model never forgets |
| HG-DAgger | A human takes over only before a failure; the takeovers become training data |
| Mean time between interventions | How long the robot runs alone before a human steps in |
| Few:many | The robot asks for help, so a few operators supervise many robots |
| Real-clock eval | Count thinking time; the real world doesn't pause |

## Open questions

- What does RL actually teach a reasoning model about manipulation, and why does shorter reasoning fail? (PEAR)
- How well do world-model and sim scores predict real-robot results? (1X, World Labs)
- Can human walking or hand video replace robot data, and how is it mapped onto robot bodies? (EgoHTR, Human-1M)
- How do you make one whole-body model steerable and compositional? (Boston Dynamics)
- How should a benchmark measure reuse, e.g. success after N real-robot trials? (RobIn lab)
- How small can System 1 get and still work at tens of Hz on the robot?
- What should flow back up from System 0 to System 2 when a grasp slips? (Hang Zhao)
- Will the deployment flywheel saturate on narrow tasks, or keep learning with a strong base? (Tedrake)
