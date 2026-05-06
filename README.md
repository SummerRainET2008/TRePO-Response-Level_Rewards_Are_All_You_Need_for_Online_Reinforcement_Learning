# 响应级奖励等效词级奖励 - 在线增强学习中的数学原理 Response-Level Rewards Are Equivalent to Token-Level Rewards: Mathematical Principles for Online Reinforcement Learning


## 总结 Summary

这个工作从基本公式推导出发，主要回答几个在线增强学习中的典型现象和问题，**中间token零奖励假设**究竟对训练算法有什么样的影响？基于这个假设的算法和使用精确的*词级奖励*的算法有什么本质区别？如火如荼的GRPO、RLOO、ReMax等REINFORCE风格的算法和PPO算法在**理论建模能力**上有什么区别？奖励衰减$\gamma$有什么影响？

Starting from first-principles derivations, this work addresses several common questions in online RL: what impact does the **zero-reward assumption for intermediate tokens** have on training algorithms? What is the essential difference between algorithms built on this assumption and systems trained with explicit **token-level rewards**? What are the differences in **theoretical modeling capacity** between REINFORCE-style methods such as GRPO, RLOO, and ReMax, and PPO? What role does reward discounting $\gamma$ play?

增强学习中的公式看起来比较复杂，但在LLM应用场景中，有一些较为特殊的地方使得给RL做一些相对精确的分析更加容易了。

RL formulas often look complicated, but LLM RL has special structure that makes relatively precise analysis easier.

## 中间token零奖励假设 Zero-Reward Assumption for Intermediate Tokens

这在一些典型的场景中非常常见，比如问答场景，奖励模型通常只对一个完整的回答进行整体质量的打分；
数学推理场景，奖励模型可以简化为一条规则，最终结果是否正确。
对中间过程人工标注和训练词级奖励模型，是一个非常消耗人力、甚至难以保证准确的过程。
因此实践当中，人们往往假设，中间生成的词语不产生直接奖励，只有最后一个词语产生整个完整句子的奖励分数，这就是**中间token零奖励假设**。

This assumption is very common in typical scenarios. In question answering, reward models usually score only the overall quality of a complete response. In mathematical reasoning, reward can be simplified to whether the final answer is correct.
Manually annotating intermediate steps and training token-level reward models is labor-intensive and can be unreliable. Therefore, in practice, people often assume intermediate ge
nerated tokens receive no direct reward, and only the last token receives the score for the whole response. This is the **zero-reward assumption for intermediate tokens**.

这个假设看上去不合理。对于人类来说，不看最后一个词语也能几乎精准地理解整句话的含义。
虽然人类也无法定量地说清楚每个词语究竟贡献了多少信息量，但它们的贡献就的的确确、安安静静地在那里，只是我们很难计算。

At first glance, this assumption seems unreasonable. For humans, the meaning of a sentence is often clear even before seeing the final token. We may not be able to quantify each t
oken's exact information contribution, but those contributions do exist; they are just hard to compute.

过去两年中几个典型的工作也先后提到了这个假设的不合理之处，
比如ReMax指出RLHF任务训练类似**单阶段**训练而非**多阶段**[1]，
GRPO指出因为中间奖励为0而值函数需要在**每一个位置精确计算**而因此复杂而不必要[2]，
而RLOO则认为人类只关心整体奖励而中间真实的奖励不存在[7]。
因此这些工作认为PPO中的值函数网络是没必要存在的。

Several representative works over the past two years discussed this concern. ReMax argued RLHF training is closer to a **single-stage** rather than **multi-stage** proce
ss [1]. GRPO argued that with zero intermediate rewards, computing value functions accurately at **every position** is complex and unnecessary [2]. RLOO argued that humans ca
re about overall reward and that intermediate ``true'' rewards do not really exist [7]. Based on this, these works suggest PPO's value network is unnecessary.

自然而然的，人们会问这么一个问题，如果这个响应级奖励可以被分解到每个问题，那么基于**中间token零奖励假设**的RL训练和基于token奖励精确训练的系统究竟差多少？

Naturally, one asks: if response-level reward can be decomposed across tokens, how different is RL trained with the **zero-reward assumption for intermediate tokens** from RL
trained with exact token-level rewards?

这种思路的代表是ABC模型[8]，采用奖励模型中最后一层的注意力机制将整体奖励进行分解，结果真的得到了提高。
他们的实验是基于非逻辑推理题目，响应级奖励模型返回一个实数。
那如果对于逻辑推理的题目、响应级奖励只返回0和1的情况呢？

A representative approach is the ABC model [8], which uses the final-layer attention of the reward model to decompose overall reward and reports gains. Their experiments use non-l
ogic reasoning tasks where response-level reward is a scalar. What about logic reasoning tasks where response-level reward is only 0/1?

咱们的工作从理论上推导下它们究竟差多少。

Our work derives the theoretical difference.

## 基本原理

Please read our articles ([zh](README.zh.pdf), [en](README.en.pdf)) and [paper 2025](https://arxiv.org/pdf/2506.02553);


## 引用 (References)
[1] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.

[2] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[3] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. arXiv preprint arXiv:2410.01679, 2024.

[4] AI Team. RL Training For Math Reasoning Introduction and Motivation. https://www.perplexity.ai/hub/blog/rl-training-for-math-reasoning. 2025.

[5] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[6]  Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, and Yi Wu. Is dpo superior to ppo for llm alignment? a comprehensive study. arXiv preprint arXiv:2404.10719, 2024.

[7] Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet ´ Ust ¨ un, ¨ and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

[8] Alex J Chan, Hao Sun, Samuel Holt, and Mihaela Van Der Schaar. Dense reward for free in reinforcement learning from human feedback. arXiv preprint arXiv:2402.00782, 2024.

[9] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. second, 2018.

[10] Weizhen Wang, Jianping He, and Xiaoming Duan. Analysis of on-policy policy gradient methods under the distribution mismatch. arXiv preprint arXiv:2503.22244, 2025.
