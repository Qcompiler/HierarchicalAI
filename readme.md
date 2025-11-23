
# 运行示例

``` source load.sh ```

``` srun -N 1  --pty --gres=gpu:4090:1  python two_layer.py --hidden 64  --epoches 6 --new 1 --round 4  ```


其中 --new 1 表示采用降低复杂度的方案训练，例如 --new 1 round 4 那么我们训练四层的网络，
依次训练第一层、第二层。。。第四层;
--new 0 表示采用原始的方案训练，例如 --new 0 round 4 那么我们训练四层的网络，一次完成训练