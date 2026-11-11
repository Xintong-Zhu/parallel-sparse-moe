"""Exercise the actual PBS measurement/validation block with interleaved streams."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PlacementLoggingTests(unittest.TestCase):
    def test_diagnostics_cannot_corrupt_trials_and_failures_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'scripts/vanda_cpu_placement.pbs').read_text()
        start = script.index('            timeout', script.index('echo "Starting $stage"'))
        end = script.index('            echo "PASS $stage"', start)
        measurement = script[start:end]
        bash = shutil.which('bash')
        if os.name == 'nt':
            bash = r'C:\Program Files\Git\bin\bash.exe'
        self.assertTrue(bash and Path(bash).exists(), 'Bash is required')
        fixture = r'''
set -e
output_dir=.
stage=fixture
log=fixture.txt
map_by=test
ranks=4
placement=la
communication=p2p
skew=moderate
layout=A
tokens=32768
timeout() { shift 3; "$@"; }
mpirun() {
  printf 'benchmark_expert_owners=0,1\n'
  printf 'benchmark_expert_histogram=4,4\n'
  printf 'benchmark_rank_assignment_histogram=4,4\n'
  for trial in 0 1 2 3 4; do
    # Merging streams would deterministically destroy the benchmark line prefix.
    printf 'diagnostic,rank=1,stage=test' >&2
    ok=1
    if [[ "$trial" == "${FAIL_TRIAL:-}" ]]; then ok=0; fi
    printf 'benchmark,trial=%s,bitwise_ok=%s\n' "$trial" "$ok"
    printf '\n' >&2
  done
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'smoke.sh').write_text(fixture + measurement, newline='\n')
            good = subprocess.run([bash, 'smoke.sh'], cwd=work, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertNotIn('diagnostic', (work / 'fixture.txt').read_text())
            self.assertEqual((work / 'fixture.stderr').read_text().count('diagnostic'), 5)
            bad = subprocess.run([bash, 'smoke.sh'], cwd=work, capture_output=True,
                                 text=True, env=dict(os.environ, FAIL_TRIAL='2'))
            self.assertNotEqual(bad.returncode, 0, 'incorrect output must still fail')


if __name__ == '__main__':
    unittest.main()
