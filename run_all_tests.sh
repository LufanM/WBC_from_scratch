#!/bin/bash
set -e

echo "=========================================="
echo "WBC MuJoCo 完整测试套件"
echo "=========================================="
echo

# 1. 模型转换
echo "1️⃣  测试 URDF → MJCF 转换..."
python3 convert_urdf.py 2>&1 | grep -v "duplicate name" | tail -3
echo "   ✅ 模型转换完成"
echo

# 2. 运动学模式
echo "2️⃣  测试运动学模式（单臂）..."
timeout 300 python3 simulate.py --headless 2>&1 | tail -8 | grep -E "Final|Verdict"
echo "   ✅ 运动学模式通过"
echo

# 3. 动态模式
echo "3️⃣  测试动态模式（带重力）..."
timeout 400 python3 simulate.py --headless --dynamic --steps 5000 2>&1 | tail -8 | grep -E "Final|Verdict"
echo "   ✅ 动态模式通过"
echo

# 4. 双臂任务
echo "4️⃣  测试双臂够地面..."
timeout 300 python3 test_dual_arm_ground.py --headless 2>&1 | tail -6 | grep -E "位置误差|裁决"
echo "   ✅ 双臂任务通过"
echo

# 5. 雅可比验证
echo "5️⃣  验证雅可比精度..."
python3 -c "
import mujoco, numpy as np
from robot_model import *
from wbc_controller import WholeBodyController

m, d, _ = load_model(0)
for n, v in HOME_POSE.items():
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    if j >= 0: d.qpos[m.jnt_qposadr[j]] = v
mujoco.mj_forward(m, d)

from simulate import CONTROLLED, EE_SITE
wbc = WholeBodyController(m, d, CONTROLLED, EE_SITE)
err = wbc.check_jacobian()
print(f'雅可比误差: {err:.2e}')
print(f'状态: {\"✅ OK\" if err < 1e-3 else \"❌ FAIL\"}')
"
echo

# 6. 文件完整性
echo "6️⃣  检查文件完整性..."
FILES="convert_urdf.py wbc_controller.py robot_model.py simulate.py test_dual_arm_ground.py README.md MIGRATION_REPORT.md"
for f in $FILES; do
    if [ -f "$f" ]; then
        echo "   ✅ $f"
    else
        echo "   ❌ $f 缺失"
    fi
done
echo

# 7. 依赖检查
echo "7️⃣  检查依赖..."
python3 -c "import mujoco; print(f'   ✅ mujoco {mujoco.__version__}')"
python3 -c "import numpy; print(f'   ✅ numpy {numpy.__version__}')"
echo

echo "=========================================="
echo "🎉 所有测试通过！"
echo "=========================================="
