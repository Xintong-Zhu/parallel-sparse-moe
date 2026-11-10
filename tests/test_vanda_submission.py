"""Exercise source provenance and early failure without a PBS installation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SubmissionTests(unittest.TestCase):
    def test_frozen_source_without_compute_git(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            (repo / "scripts").mkdir(parents=True)
            for name in ("submit_vanda_cpu_formal.sh", "vanda_cpu_formal.pbs",
                         "vanda_cpu_compare.pbs"):
                shutil.copy(project / "scripts" / name, repo / "scripts" / name)
            (repo / "payload").write_text("original\n")
            def run(*args, env=None):
                return subprocess.run(args, cwd=repo, env=env, text=True, capture_output=True)
            for args in (("init",), ("config", "user.name", "Test"),
                         ("config", "user.email", "test@example.invalid"),
                         ("add", "."), ("commit", "-m", "fixture")):
                self.assertEqual(run("git", *args).returncode, 0)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            qsub = fake_bin / "qsub"
            qsub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_ARGS"\n')
            qsub.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
                       TEST_ARGS=str(root / "qsub_args"))
            result = run("bash", "scripts/submit_vanda_cpu_formal.sh", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = (root / "qsub_args").read_text().splitlines()
            self.assertEqual(arguments[0], "-v")
            snapshot = Path(arguments[1].removeprefix("MOE_SOURCE_DIR="))
            self.assertEqual((snapshot / ".moe-source-commit").read_text().strip(),
                             run("git", "rev-parse", "HEAD").stdout.strip())
            # Change the checkout after submission: the frozen tree must survive.
            (repo / "payload").write_text("changed\n")
            self.assertEqual((snapshot / "payload").read_text(), "original\n")
            nodefile = root / "nodes"
            nodefile.write_text("one-node\n")
            startup = root / "startup.sh"
            startup.write_text('module() { :; }\ngit() { echo git-unavailable >&2; return 127; }\n')
            env.update(BASH_ENV=str(startup), PBS_O_WORKDIR=str(repo), PBS_JOBID="test.server",
                       PBS_NODEFILE=str(nodefile), MOE_SOURCE_DIR=str(snapshot))
            result = run("bash", str(snapshot / "scripts/vanda_cpu_formal.pbs"), env=env)
            self.assertNotEqual(result.returncode, 0)
            status = repo / "private/formal-cpu-test/job_status.txt"
            self.assertIn("stage=allocation", status.read_text())
            self.assertNotIn("git-unavailable", result.stdout + result.stderr)
            # A modified snapshot must fail before any allocation or MPI work.
            (snapshot / "payload").write_text("tampered\n")
            result = run("bash", str(snapshot / "scripts/vanda_cpu_formal.pbs"), env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stage=source_verification", status.read_text())
            env.pop("MOE_SOURCE_DIR")
            result = run("bash", "scripts/vanda_cpu_formal.pbs", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stage=source_verification", status.read_text())
            self.assertIn("git-unavailable", result.stdout + result.stderr)
            env.pop("BASH_ENV")
            result = run("bash", "scripts/submit_vanda_cpu_formal.sh",
                         "scripts/vanda_cpu_compare.pbs", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "qsub_args").read_text().splitlines()[-1].endswith(
                "scripts/vanda_cpu_compare.pbs"))


if __name__ == "__main__":
    unittest.main()
