/**
 * 课程 3.5: CUDA 线程组织模型详解 - 二维网格二维线程块
 * 文件名: grid2D_block2D.cu
 * 作者: 权 双
 * 日期: 2023-08-14
 * 功能: 使用二维网格和二维线程块组织模型计算二维矩阵加法
 *
 * 线程组织模型介绍：
 * 1. 二维网格 (2D Grid) - 线程块在网格中按二维排列
 * 2. 二维线程块 (2D Block) - 线程在线程块中按二维排列
 * 3. 二维矩阵映射 - 直观的坐标对应关系
 *
 * 优势：
 * - 代码直观易懂，坐标映射清晰
 * - 适合图像处理、矩阵运算等二维数据
 * - 利用 GPU 的二维线程组织特性
 *
 * 学习目标：
 * - 掌握二维线程组织的索引计算
 * - 理解网格和线程块的维度配置
 * - 学习边界检查的重要性
 * - 体验多维并行计算的优势
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * GPU 内核函数 - 二维矩阵加法
 *
 * @param A: 输入矩阵 A
 * @param B: 输入矩阵 B
 * @param C: 输出矩阵 C (C = A + B)
 * @param nx: 矩阵宽度（列数）
 * @param ny: 矩阵高度（行数）
 *
 * 线程索引计算解析：
 * - threadIdx.x, threadIdx.y: 线程在线程块内的坐标
 * - blockIdx.x, blockIdx.y: 线程块在网格中的坐标
 * - blockDim.x, blockDim.y: 线程块的维度大小
 *
 * 二维到一维映射：
 * idx = iy * nx + ix (行优先存储)
 */
__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    /**
     * 计算当前线程对应的二维坐标
     *
     * ix 计算 (X坐标)：
     * - threadIdx.x: 线程在块内的 X 坐标 (0 到 blockDim.x-1)
     * - blockIdx.x: 块在网格中的 X 坐标 (0 到 gridDim.x-1)
     * - blockDim.x: 线程块在 X 方向的大小
     * - ix = threadIdx.x + blockIdx.x * blockDim.x
     *
     * iy 计算 (Y坐标)：
     * - threadIdx.y: 线程在块内的 Y 坐标 (0 到 blockDim.y-1)
     * - blockIdx.y: 块在网格中的 Y 坐标 (0 到 gridDim.y-1)
     * - blockDim.y: 线程块在 Y 方向的大小
     * - iy = threadIdx.y + blockIdx.y * blockDim.y
     */
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  // 全局 X 坐标
    int iy = threadIdx.y + blockIdx.y * blockDim.y;  // 全局 Y 坐标

    /**
     * 将二维坐标转换为一维数组索引
     *
     * 行优先存储 (Row-major order)：
     * - 第 iy 行，第 ix 列的元素
     * - 索引 = 行号 × 列数 + 列号
     * - idx = iy * nx + ix
     *
     * 示例 (nx=4, ny=3):
     * (0,0)→0  (0,1)→1  (0,2)→2  (0,3)→3
     * (1,0)→4  (1,1)→5  (1,2)→6  (1,3)→7
     * (2,0)→8  (2,1)→9  (2,2)→10 (2,3)→11
     */
    unsigned int idx = iy * nx + ix;

    /**
     * 边界检查 - 防止数组越界
     *
     * 为什么需要边界检查？
     * - 网格大小通常向上取整，可能超出矩阵尺寸
     * - 例如：矩阵 16×8，线程块 4×4，网格 4×2 = 32 个线程块
     * - 总线程数：4×4×4×2 = 128，但矩阵只有 16×8 = 128 个元素
     * - 当线程块不能整除矩阵维度时，会有多余线程
     *
     * 检查条件：
     * - ix < nx: X 坐标不超出矩阵宽度
     * - iy < ny: Y 坐标不超出矩阵高度
     */
    if (ix < nx && iy < ny)
    {
        /**
         * 执行矩阵加法
         * C[idx] = A[idx] + B[idx]
         *
         * 只有在有效范围内的线程才执行计算
         * 避免访问未分配的内存区域
         */
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * 主函数 - 二维网格二维线程块矩阵加法演示
 */
int main(void)
{
    /**
     * 第一步：GPU 设备初始化
     * 设置要使用的 GPU 设备（默认设备 0）
     */
    setGPU();

    /**
     * 第二步：定义矩阵参数和分配内存
     *
     * 矩阵配置：
     * - nx = 16: 矩阵宽度（列数）
     * - ny = 8:  矩阵高度（行数）
     * - nxy = 128: 矩阵总元素数
     * - 总内存：128 × 4 = 512 字节（每个 int 4 字节）
     */
    int nx = 16;        // 矩阵宽度
    int ny = 8;         // 矩阵高度
    int nxy = nx * ny;  // 总元素数：16 × 8 = 128
    size_t stBytesCount = nxy * sizeof(int);  // 总字节数：128 × 4 = 512

    /**
     * 第三步：分配和初始化主机内存
     *
     * 内存布局：
     * - ipHost_A: 输入矩阵 A
     * - ipHost_B: 输入矩阵 B
     * - ipHost_C: 输出矩阵 C（用于存储 GPU 计算结果）
     */
    int *ipHost_A, *ipHost_B, *ipHost_C;
    ipHost_A = (int *)malloc(stBytesCount);
    ipHost_B = (int *)malloc(stBytesCount);
    ipHost_C = (int *)malloc(stBytesCount);

    // 检查内存分配是否成功
    if (ipHost_A != NULL && ipHost_B != NULL && ipHost_C != NULL)
    {
        /**
         * 初始化输入数据
         *
         * 数据模式：
         * - A[i] = i       (0, 1, 2, 3, ..., 127)
         * - B[i] = i + 1   (1, 2, 3, 4, ..., 128)
         * - 预期结果 C[i] = A[i] + B[i] = 2i + 1
         *
         * 这种简单的数据模式便于验证计算结果的正确性
         */
        for (int i = 0; i < nxy; i++)
        {
            ipHost_A[i] = i;        // A: 0, 1, 2, 3, ...
            ipHost_B[i] = i + 1;    // B: 1, 2, 3, 4, ...
        }
        memset(ipHost_C, 0, stBytesCount);  // C 初始化为 0
    }
    else
    {
        printf("主机内存分配失败!\n");
        exit(-1);
    }

    /**
     * 第四步：分配和初始化设备内存
     *
     * GPU 内存管理：
     * - cudaMalloc: 在 GPU 全局内存中分配空间
     * - cudaMemcpy: 在主机和设备间传输数据
     * - ErrorCheck: 检查 CUDA API 调用是否成功
     */
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;

    // 分配设备内存
    ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_B, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_C, stBytesCount), __FILE__, __LINE__);

    // 检查设备内存分配并传输数据
    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C != NULL)
    {
        // 将输入数据从主机复制到设备
        ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }
    else
    {
        printf("设备内存分配失败\n");
        // 清理已分配的主机内存
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(1);
    }

    /**
     * 第五步：配置 GPU 内核启动参数
     *
     * 二维配置详解：
     *
     * 线程块配置 block(4, 4)：
     * - blockDim.x = 4, blockDim.y = 4
     * - 每个线程块包含 4×4 = 16 个线程
     * - 线程在线程块内按二维排列：
     *   (0,0) (1,0) (2,0) (3,0)
     *   (0,1) (1,1) (2,1) (3,1)
     *   (0,2) (1,2) (2,2) (3,2)
     *   (0,3) (1,3) (2,3) (3,3)
     *
     * 网格配置 grid(4, 2)：
     * - gridDim.x = (16+4-1)/4 = 4
     * - gridDim.y = (8+4-1)/4 = 2
     * - 网格包含 4×2 = 8 个线程块
     * - 线程块在网格中按二维排列：
     *   Block(0,0) Block(1,0) Block(2,0) Block(3,0)
     *   Block(0,1) Block(1,1) Block(2,1) Block(3,1)
     *
     * 总线程数：8 个线程块 × 16 个线程/块 = 128 个线程
     * 正好等于矩阵元素数 16×8 = 128
     */
    dim3 block(4, 4);  // 线程块：4×4 = 16 个线程
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);  // 网格：4×2 = 8 个线程块

    printf("线程配置：网格:<%d, %d>, 线程块:<%d, %d>\n", grid.x, grid.y, block.x, block.y);
    printf("总线程数：%d 个线程块 × %d 个线程/块 = %d 个线程\n",
           grid.x * grid.y, block.x * block.y, grid.x * grid.y * block.x * block.y);
    printf("矩阵规模：%d × %d = %d 个元素\n", nx, ny, nxy);

    /**
     * 第六步：启动 GPU 内核
     *
     * 内核启动语法：
     * kernelName<<<grid, block>>>(参数列表)
     *
     * 执行模式：
     * - 异步启动：函数立即返回，不等待 GPU 完成
     * - 并行执行：128 个线程同时执行矩阵加法
     * - 每个线程处理一个矩阵元素
     */
    addMatrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);

    /**
     * 第七步：获取计算结果
     *
     * 数据传输：
     * - 将结果从设备内存复制回主机内存
     * - cudaMemcpy 会隐式同步，等待 GPU 计算完成
     */
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    /**
     * 第八步：验证计算结果
     *
     * 显示前 10 个元素的计算结果：
     * - 输入 A[i] = i
     * - 输入 B[i] = i + 1
     * - 预期结果 C[i] = A[i] + B[i] = 2i + 1
     *
     * 验证示例：
     * i=0: A=0, B=1, C=1 (0+1=1) ✓
     * i=1: A=1, B=2, C=3 (1+2=3) ✓
     * i=2: A=2, B=3, C=5 (2+3=5) ✓
     */
    printf("\n计算结果验证（前10个元素）：\n");
    for (int i = 0; i < 10; i++)
    {
        printf("id=%d, A=%d, B=%d, C=%d (预期:%d)\n",
               i + 1, ipHost_A[i], ipHost_B[i], ipHost_C[i], 2 * i + 1);
    }

    /**
     * 第九步：清理资源
     *
     * 内存管理最佳实践：
     * - 释放主机内存：free()
     * - 释放设备内存：cudaFree()
     * - 重置设备状态：cudaDeviceReset()
     * - 对称分配释放，避免内存泄漏
     */
    // 释放主机内存
    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);

    // 释放设备内存
    ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__);

    // 重置 CUDA 设备
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    printf("\n程序执行完成，内存资源已清理\n");
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 二维线程组织模型：
 *    - 网格维度：gridDim.x × gridDim.y
 *    - 线程块维度：blockDim.x × blockDim.y
 *    - 线程坐标：(threadIdx.x, threadIdx.y)
 *    - 线程块坐标：(blockIdx.x, blockIdx.y)
 *
 * 2. 索引计算公式：
 *    - 全局坐标：ix = threadIdx.x + blockIdx.x * blockDim.x
 *    - 全局坐标：iy = threadIdx.y + blockIdx.y * blockDim.y
 *    - 一维索引：idx = iy * nx + ix
 *
 * 3. 网格大小计算：
 *    - X 方向：(nx + blockDim.x - 1) / blockDim.x
 *    - Y 方向：(ny + blockDim.y - 1) / blockDim.y
 *    - 向上取整确保覆盖所有元素
 *
 * 4. 边界检查的重要性：
 *    - 防止访问越界内存
 *    - 处理非整除的矩阵尺寸
 *    - 确保程序稳定性
 *
 * 5. 性能考虑：
 *    - 线程块大小影响占用率
 *    - 二维组织适合二维数据
 *    - 内存合并访问模式
 *    - 避免分支发散
 *
 * 6. 实际应用场景：
 *    - 图像处理：像素级并行操作
 *    - 矩阵运算：线性代数计算
 *    - 科学计算：网格数值方法
 *    - 深度学习：卷积神经网络
 *
 * 7. 调试技巧：
 *    - 使用小规模数据测试
 *    - 验证索引计算正确性
 *    - 检查边界条件处理
 *    - 比对 CPU 参考结果
 */
