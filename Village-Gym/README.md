---


---

<h1 id="welcome-to-village-game">Welcome to Village Game!</h1>
<p>There are many applications that one can build by applying Reinforcement Learning (RL). We have chosen to develop a system, popularly called “Village Game” that will teach a computer agent on how to play this educational game to maximize the reward which in this case the total bank balance for the agent.</p>
<p>In developing countries there are serious problems in various areas mainly due to lack of education. We think providing practical education in a game environment can promote social equality and can have a profound effect especially when done at a young age. This game when developed completely can teach the children the responsibilities of a family and how various decisions can affect their livelihood.</p>
<p>The solution of this problem uses Reinforcement Learning using Q-Learning technique in unknown environments where the rewards are time delayed. The game also raises awareness of the importance of decision-making, based on objective data and the assessment of the consequences of various short, medium and long-term action alternatives.</p>
<p>In this problem the agent doesn’t know how the real world works. The agent will have to act in the real world, learn how it works from experiencing it and trying to take actions that yield high rewards. This problem combines our dual objective of an area that we want to understand the various aspects of Q-Learning technique and at the same time create a positive social impact.</p>
<p>Our intention is to implement the game using Pygame, a popular Python library for coding video games. And then implement the AI solution in Python. We intend to publish the details of our implementation, testing and analysis through repeated gameplay.</p>
<p>For more information <a href="https://github.com/rakeshtl/artificial_intelligence/blob/master/Village-Gym/Village%20Game%20Description.pdf">read here</a></p>
<p>For our first version <a href="https://github.com/rakeshtl/artificial_intelligence/blob/master/Village-Gym/Village%20Game%20Version%201%20Report.pdf">read here</a></p>
<h1 id="install">Install</h1>
<p><strong>First Install Gym</strong><br>
pip install --user gym</p>
<p>If you already have it then upgrade to latest version<br>
pip install --upgrade pip</p>
<p><strong>Next Install Village game</strong><br>
Go to directory where <a href="http://setup.py">setup.py</a> is located.<br>
cd Village-Gym/gym-village<br>
pip install -e .</p>
<h2 id="unit-testing">Unit Testing</h2>
<p>Go to directory Village-Gym/Tests</p>
<p>cd Village-Gym/Tests<br>
python <a href="http://VillageTest.py">VillageTest.py</a></p>
<h2 id="learning">Learning</h2>
<p>python td_agent2.py learn</p>
<h2 id="play">Play</h2>
<p>python td_agent2.py play</p>

