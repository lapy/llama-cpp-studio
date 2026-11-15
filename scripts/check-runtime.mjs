import { execFileSync } from "node:child_process";

const requiredNodeMajor = 24;
const requiredPython = [3, 12];

function fail(message) {
  console.error(`runtime check failed: ${message}`);
  process.exitCode = 1;
}

const nodeVersion = process.versions.node;
const nodeMajor = Number(nodeVersion.split(".")[0]);
if (nodeMajor !== requiredNodeMajor) {
  fail(`Node.js 24 is required; found ${nodeVersion}`);
} else {
  let pythonVersion;
  try {
    pythonVersion = execFileSync(
      "python",
      ["-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
      { encoding: "utf8" },
    ).trim();
  } catch {
    fail("Python 3.12 is required, but `python` could not be executed");
  }

  if (pythonVersion) {
    const [pythonMajor, pythonMinor] = pythonVersion.split(".").map(Number);
    if (pythonMajor !== requiredPython[0] || pythonMinor !== requiredPython[1]) {
      fail(`Python 3.12 is required; found ${pythonVersion}`);
    } else {
      console.log(`supported runtimes: Node.js ${nodeVersion}, Python ${pythonVersion}`);
    }
  }
}
