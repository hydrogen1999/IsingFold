from types import SimpleNamespace
from collections import Counter
import cProfile,pstats,time
import dwave_networkx as dnx,networkx as nx,numpy as np,torch
from isingfold.embedding import LogicalProblem
from constructor_features import ConstructorFeatureContext,WIDTH
from constructor_rollout import episode,EmbeddingEnv
from layout_policy import LayoutActorCritic

torch.set_num_threads(1);torch.manual_seed(0)
h=dnx.pegasus_graph(3);l=nx.cycle_graph(20)
t=SimpleNamespace(host=h,logical=l,problem=LogicalProblem.from_dicts({v:0 for v in l},{e:-1 for e in l.edges()}),name='pegasus3-profiling',lineage='toy-profile',ground_energy=-20)
class FC(ConstructorFeatureContext):
 def __init__(self,t):super().__init__(t);self.seconds=0.;self.rows=0
 def observe(self,*a,**kw):
  st=time.perf_counter();r=super().observe(*a,**kw);self.seconds+=time.perf_counter()-st;self.rows+=1;return r
fc=FC(t);model=LayoutActorCritic(64,WIDTH);hist=[];orig=EmbeddingEnv.step
def step(self,d,i,**kw):
 legal=[c for c,ok in zip(d.candidates,d.legal_mask) if ok]
 kind=lambda c:c.provenance.split(':')[0] if c.opcode.value=='REWRITE_ONE' else c.opcode.value
 hist.append({'support':dict(Counter(kind(c) for c in legal)),'chosen':kind(d.candidates[i])})
 return orig(self,d,i,**kw)
EmbeddingEnv.step=step;p=cProfile.Profile();p.enable();r=episode(t,model,fc,1.,5,np.random.default_rng(0),20.,train=True,objective='feasibility',reward_reads=8);p.disable();st=pstats.Stats(p)
print({'host_qubits':len(h),'logical_variables':len(l),'steps':r['steps'],'reason':r['reason'],'total_profiled_seconds':round(r['secs'],4),'constructor_feature_seconds':round(fc.seconds,4),'rows':fc.rows,'actions':hist})
for name in ['_prepare','build_observation','_assert_search_state','distribution_value','_local','_summary','generate']:
 entries=[(k[0].split('/')[-1],v[1],round(v[3],4)) for k,v in st.stats.items() if k[2]==name]
 print(name,entries)
