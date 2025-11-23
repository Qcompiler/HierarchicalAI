import vector_add  # 导入编译好的模块
import numpy as np

# 生成测试数据（float32 类型，确保与 C++ 类型匹配）
a = np.random.rand(1000).astype(np.float32)
b = np.random.rand(1000).astype(np.float32)

# 调用 ARM Neon 加速的向量加法
c = vector_add.add(a, b)

# 验证结果（与 numpy 原生加法对比）
c_numpy = a + b
assert np.allclose(c, c_numpy, atol=1e-6), "结果不一致！"

print("测试通过！")
print(f"示例：a[0] = {a[0]:.4f}, b[0] = {b[0]:.4f}, c[0] = {c[0]:.4f}")