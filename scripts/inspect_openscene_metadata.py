import tarfile,pickle,io,numpy as np,json,importlib
from pathlib import Path
class Safe(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','_reconstruct'),('numpy._core.multiarray','scalar'),('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer')}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Blocked global '+module+'.'+name)
        return getattr(importlib.import_module(module),name)
p=Path('data/public/openscene-mini-audit/openscene_metadata_mini.tgz')
with tarfile.open(p,'r|gz') as t:
    for i,m in enumerate(t):
        print(m.name,m.size,flush=True)
        if m.isfile() and m.name.endswith('.pkl'):
            assert m.size<256*1024**2
            obj=Safe(io.BytesIO(t.extractfile(m).read())).load();print('TYPE',type(obj),'LEN',len(obj));o=obj[0] if isinstance(obj,list) else obj;print('SCHEMA',{k:(type(v).__name__,str(v.shape) if isinstance(v,np.ndarray) else '') for k,v in o.items()});print('VALUES',{k:v for k,v in o.items() if 'vel' in k or 'time' in k or 'ego' in k});break
