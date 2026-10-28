"""Wait for the fixed experiment pipeline, then validate and build its report."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from mini_wa_jepa.provenance import write_json
from training.flow_matching import file_sha256


def audit(root=Path('artifacts/restart_stage7')):
    review=root/'codex_review';state=dict(completed=False,current='waiting_for_experiments')
    def update(**kw):
        state.update(kw);write_json(root/'delivery_status.json',state)
    def run(name,argv):
        update(current=name)
        log=review/f'{name}.log'
        with log.open('w') as stream:
            result=subprocess.run(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed ({result.returncode}); see {log}')
        return dict(command=list(map(str,argv)),returncode=result.returncode,log=str(log))
    update()
    try:
        while True:
            status=json.loads((root/'remaining_status.json').read_text())
            if status.get('error'):raise RuntimeError('Experiment pipeline stopped; inspect remaining_status.json')
            if status.get('completed'):break
            time.sleep(5)
        # Refresh browser checks against formal recordings, retaining the smoke proof.
        old=review/'browser_checks.json'
        if old.exists() and not (review/'browser_checks_smoke.json').exists():
            (review/'browser_checks_smoke.json').write_bytes(old.read_bytes())
        run('browser_formal',['node',review/'check_browser.cjs'])
        update(current='unittest')
        import unittest
        started=time.perf_counter()
        with (review/'unittest.log').open('w') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
        write_json(review/'tests_result.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,seconds=time.perf_counter()-started))
        if not result.wasSuccessful():raise RuntimeError('Test failure; preserved unittest.log')
        checks=run('marimo_check',['.venv/bin/marimo','check','app.py',
            'chapters/chapter7_candidate_planning/app.py','scripts/dashboard_marimo.py'])
        write_json(review/'marimo_checks.json',dict(passed=True,check=checks))
        update(current='rrd_native_readback');rows=[]
        # Each original stream is read with its installed native SDK CLI, without a viewer.
        for i,path in enumerate(sorted(root.rglob('*.rrd'))):
            proof=run(f'rrd_{i:03d}',[sys.executable,'-m','rerun','rrd','print',path])
            rows.append(dict(path=str(path),sha256=file_sha256(path),bytes=path.stat().st_size,**proof))
        write_json(review/'rrd_checks.json',dict(passed=bool(rows),files=len(rows),checks=rows))
        run('report',[sys.executable,'-m','scripts.report_stage7','--root',root])
        exports=[]
        for name,source in [('chapter7','chapters/chapter7_candidate_planning/app.py'),
                            ('dashboard','scripts/dashboard_marimo.py')]:
            output=root/f'{name}.html'
            if output.exists():raise FileExistsError(f'Preserve existing export: {output}')
            proof=run(f'export_{name}',['.venv/bin/marimo','export','html',source,'-o',output])
            exports.append(dict(path=str(output),bytes=output.stat().st_size,sha256=file_sha256(output),**proof))
        write_json(review/'html_exports.json',dict(passed=True,exports=exports))
        run('report_final',[sys.executable,'-m','scripts.report_stage7','--root',root])
        update(completed=True,current='delivery_complete')
    except BaseException:
        update(error=traceback.format_exc());raise


if __name__=='__main__':audit()
