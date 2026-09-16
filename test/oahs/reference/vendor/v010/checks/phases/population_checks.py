from examples import *
from dataclasses import replace
from pathlib import Path
import json
rows=[]
for n in [1,2,4,8]:
 p=profile(n)
 a=analyze(linear([producer(p),consumer(p)]),PhaseInterface(p,True))
 b=analyze(loop([producer(p),consumer(p),Action('fence','F')]),PhaseInterface(p,False))
 assert a['accepted'] and b['accepted']
 rows.append({'blocks':n,'exact_single_pair':a,'safety_loop':b})
p=profile(2)
p2=Profile(p.engines,list(p.cells.values())+[Cell('v0','ACC',1024,512),Cell('v1','ACC',1536,512)],
           list(p.groups.values())+[Group('other',('v0','v1'))],p.keys)
wrong=replace(consumer(p),group='other',blocks=('v0','v1'))
result=analyze(linear([producer(p),wrong]),PhaseInterface(p2,False))
assert result.get('kind')=='resource_balance'
Path(__file__).with_name('population_results.json').write_text(json.dumps({'block_population':rows,'independent_resource_negative':result},indent=2)+'\n')
print('profile population checks passed')
