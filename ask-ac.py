import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from actor_critic import ActorCritic

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class AskActorCritic(nn.Module):
    def __init__(self):
        super(AskActorCritic, self).__init__()
        self.data = []
        self.human_data = []

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,3) # 原始的两个action加上一个human action 默认为human action的index 为2（也就是最后一个）
        self.fc_v = nn.Linear(256,1)
        self.loss_cel = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x): # Critic
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def put_human_data(self, transition):
        self.human_data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def make_human_batch(self):
        s_lst, a_lst = [],[]
        for transition in self.human_data:
            s, a = transition
            s_lst.append(s)
            a_lst.append(a)
        s_batch,a_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst)
        self.human_data = []
        return s_batch, a_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        
        ac_loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach()) # 负的，增加“loss”
        ac_loss = ac_loss.mean()
        # 计算human loss
        s, a = self.make_human_batch()
        # print(s.size())
        if s.size() == torch.Size([0]):
            human_loss = 0
        else: 
            pi = self.pi(s, softmax_dim=1)
            # pi_a = pi.index_select(0,s)
            
            human_loss = self.loss_cel(pi, a).mean()
        loss = ac_loss + human_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
            


def main():  
    human_model = ActorCritic()
    human_model.load_state_dict(torch.load('ac_para.pkl'))

    env = gym.make('CartPole-v1')
    model = AskActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()
        step,ask_step = 0,0
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                if a == 2: # human action
                    prob = human_model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    model.put_human_data((s, a))
                    ask_step += 1
                
                s_prime, r, done, info = env.step(a)

                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                step += 1
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, ask rate : {:.2f}".format(n_epi, score/print_interval, ask_step/step))
            score = 0.0

if __name__ == '__main__':
    main()