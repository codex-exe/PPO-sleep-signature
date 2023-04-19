# PPO-sleep-signature
Sleep plays a critical role in our overall health and well-being. But many people
struggle with poor sleep quality and quantity, which can negatively impact their
physical and mental health, as well as their daily functioning and quality of life.
Improving sleep-related behaviors, such as bedtime, caffeine intake, and exposure
to light, can help to improve sleep quality. However, determining the optimal
sleep-related actions can be challenging, as different individuals have different
sleep requirements and preferences.

- We took up the Multilevel Monitoring of Activity and Sleep in Healthy
People (MMASH) dataset for the project, which offered psychological data
(such as anxiety status, stress events, and emotions) for 22 healthy
participants as well as 24-hour continuous beat-to-beat heart data, triaxial
accelerometer data, sleep quality, physical activity, and sleep duration.

- We used the Proximal Policy Optimization (PPO) algorithm to optimize
each user’s sleep using the MMASH dataset.

### What is PPO?
- PPO is a policy-based reinforcement learning algorithm, which means that it
is well-suited for problems where the optimal action is not known a priori. In
the case of sleep optimization, the optimal sleep schedule is not known, and
the algorithm must learn it through trial and error.

- PPO is designed to handle continuous action spaces, which is important for
sleep optimization since the action space consists of continuous variables
such as the time to go to bed and wake up.

- PPO is an on-policy algorithm, which means that it learns from the data that
it generates during training. This is important for sleep optimization since
the algorithm needs to explore the sleep schedule space and generate new
data in order to learn the optimal sleep schedule.

- PPO is an on-policy algorithm, which means that it learns from the data that
it generates during training. This is important for sleep optimization since
the algorithm needs to explore the sleep schedule space and generate new
data in order to learn the optimal sleep schedule.

### Contributers:
- [Alabhya Sharma]
- [Ritika Lakshminarayanan]

Colab link: https://colab.research.google.com/drive/1933n5ZR5EPRYC4kyUX1cXXUpVtxqxqFN?usp=sharing


### Architecture:
[![architecture](https://github.com/codex-exe/PPO-sleep-signature/blob/main/Results/architecture.png)

### Result Graphs:
[![result1](https://github.com/codex-exe/PPO-sleep-signature/blob/main/Results/User12%2C%20reward%20space.png)
[![result2](https://github.com/codex-exe/PPO-sleep-signature/blob/main/Results/user12_optimal.png)
