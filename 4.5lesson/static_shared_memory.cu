/**
 * 课程 4.5: CUDA 静态共享内存详解
 * 文件名: static_shared_memory.cu
 * 作者: 权 双
 * 日期: 2023-12-26
 * 功能: 静态共享内存使用演示
 *
 * 共享内存 (Shared Memory) 特点：
 * 1. 高速访问 - 接近寄存器的访问速度
 * 2. 线程块内共享 - 同一线程块内的所有线程可以访问
 * 3. 容量有限 - 每个SM的共享内存容量有限（通常48KB-164KB）
 * 4. 手动管理 - 需要程序员显式分配和同步
 * 5. 延迟低 - 比全局内存快约100倍
 *
 * 静态共享内存 vs 动态共享内存：
 * - 静态：编译时确定大小，使用 __shared__ 声明
 * - 动态：运行时确定大小，内核启动时指定
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

/**
 * GPU 内核函数 - 演示静态共享内存的使用
 *
 * @param d_A: 全局内存中的输入数组
 * @param N: 数组元素总数
 *
 * 本内核演示了静态共享内存的典型使用模式：
 * 1. 从全局内存加载数据到共享内存
 * 2. 线程同步确保所有数据加载完成
 * 3. 处理共享内存中的数据
 * 4. 输出处理结果
 */
__global__ void kernel_1(float* d_A, const int N)
{
    /**
     * 获取线程标识符
     */
    const int tid = threadIdx.x;  // 线程在块内的索引 (0-31)
    const int bid = blockIdx.x;   // 线程块索引
    const int n = bid * blockDim.x + tid;  // 全局线程索引

    /**
     * 静态共享内存声明
     *
     * __shared__ 关键字特点：
     * - 必须在内核函数内部声明
     * - 大小在编译时确定（这里是32个float）
     * - 生存期：整个线程块执行期间
     * - 作用域：当前线程块内的所有线程
     * - 初始化：不会自动初始化，包含随机值
     *
     * 内存布局：32个浮点数 = 32 * 4 = 128 字节
     */
    __shared__ float s_array[32];

    /**
     * 第一阶段：协作加载数据到共享内存
     *
     * 关键概念：合并访问 (Coalesced Access)
     * - 相邻线程访问相邻内存位置
     * - 最大化内存带宽利用率
     * - 每个线程负责加载一个元素
     */
    if (n < N)
    {
        /**
         * 从全局内存加载到共享内存
         *
         * 访问模式分析：
         * - tid=0: s_array[0] = d_A[bid*32 + 0]
         * - tid=1: s_array[1] = d_A[bid*32 + 1]
         * - ...
         * - tid=31: s_array[31] = d_A[bid*32 + 31]
         *
         * 这种模式确保了对全局内存的合并访问
         */
        s_array[tid] = d_A[n];
    }

    /**
     * 线程同步 - 关键步骤
     *
     * __syncthreads() 的作用：
     * - 阻塞所有线程，直到线程块内所有线程都到达此点
     * - 确保所有数据都已加载到共享内存
     * - 避免数据竞争 (Data Race)
     * - 类似于线程间的屏障 (Barrier)
     *
     * 为什么需要同步？
     * - 线程执行可能不同步
     * - 某些线程可能先执行完加载，某些线程可能还在加载
     * - 没有同步的话，后续访问可能读到未初始化的数据
     */
    __syncthreads();

    /**
     * 第二阶段：处理共享内存中的数据
     *
     * 策略：让线程0负责输出
     * - 避免多个线程重复输出
     * - 展示共享内存的"共享"特性
     * - 一个线程可以访问其他线程加载的数据
     */
    if (tid == 0)
    {
        /**
         * 遍历整个共享内存数组
         *
         * 重要观察：
         * - 线程0可以访问所有32个元素
         * - 这些元素是由不同线程（tid=0到31）加载的
         * - 这展示了共享内存的核心特性：线程块内共享
         */
        for (int i = 0; i < 32; ++i)
        {
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
        }
    }
}

/**
 * 主函数 - 演示静态共享内存的完整使用流程
 */
int main(int argc, char **argv)
{
    /**
     * 第一步：获取GPU设备信息
     */
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    /**
     * 输出重要的共享内存信息
     */
    std::cout << "每个线程块的共享内存大小: "
              << deviceProps.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "每个SM的共享内存大小: "
              << deviceProps.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;

    /**
     * 第二步：准备测试数据
     *
     * 数据设计：
     * - 64个元素 = 2个线程块 × 32个线程/块
     * - 每个线程块处理32个连续元素
     * - 便于观察共享内存的分块特性
     */
    int nElems = 64;
    int nbytes = nElems * sizeof(float);

    /**
     * 分配主机内存并初始化
     */
    float* h_A = nullptr;
    h_A = (float*)malloc(nbytes);

    // 初始化为连续整数，便于验证结果
    for (int i = 0; i < nElems; ++i)
    {
        h_A[i] = float(i);
    }

    std::cout << "\n初始化数据 (前16个元素): ";
    for (int i = 0; i < 16; ++i) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "..." << std::endl;

    /**
     * 第三步：分配设备内存并传输数据
     */
    float* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));

    /**
     * 第四步：配置内核启动参数
     *
     * 关键配置解析：
     * - dim3 block(32): 每个线程块32个线程
     * - dim3 grid(2): 2个线程块
     * - 总线程数: 2 × 32 = 64，正好等于数据元素数
     *
     * 共享内存使用：
     * - 每个线程块使用 32 × 4 = 128 字节共享内存
     * - 2个线程块总共使用 256 字节共享内存
     * - 远小于GPU的共享内存容量限制
     */
    dim3 block(32);
    dim3 grid(2);

    std::cout << "\n内核配置:" << std::endl;
    std::cout << "线程块大小: " << block.x << std::endl;
    std::cout << "网格大小: " << grid.x << std::endl;
    std::cout << "每个线程块的共享内存使用: " << 32 * sizeof(float) << " 字节" << std::endl;

    /**
     * 第五步：启动内核函数
     *
     * 预期行为：
     * - 线程块0处理元素0-31，输出这些值
     * - 线程块1处理元素32-63，输出这些值
     * - 每个线程块的线程0负责输出该块的所有32个元素
     */
    std::cout << "\n启动内核，输出结果:" << std::endl;
    kernel_1<<<grid, block>>>(d_A, nElems);

    /**
     * 等待内核完成，确保所有printf输出都显示
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    /**
     * 第六步：清理资源
     */
    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 静态共享内存的声明和使用：
 *    - 使用 __shared__ 关键字
 *    - 编译时确定大小
 *    - 线程块内所有线程共享
 *    - 需要手动同步访问
 *
 * 2. 典型使用模式：
 *    - 加载：从全局内存到共享内存
 *    - 同步：__syncthreads() 确保数据一致性
 *    - 处理：在共享内存中进行快速计算
 *    - 输出：将结果写回全局内存或输出
 *
 * 3. 性能优势：
 *    - 访问延迟：~1-2个时钟周期 vs 全局内存的~400-800个周期
 *    - 带宽：非常高的内部带宽
 *    - 能耗：比全局内存访问节能
 *
 * 4. 使用场景：
 *    - 数据重用：多个线程需要访问相同数据
 *    - 线程协作：需要线程间交换数据
 *    - 算法优化：矩阵乘法、卷积、归约等
 *
 * 5. 注意事项：
 *    - 容量限制：每个SM的共享内存有限
 *    - 银行冲突：避免多个线程同时访问同一银行
 *    - 同步开销：__syncthreads() 有性能开销
 *    - 占用率影响：共享内存使用会影响线程块并发数
 *
 * 6. 与其他内存的对比：
 *    - 寄存器：更快但私有，容量极小
 *    - 共享内存：快速且共享，容量适中
 *    - 全局内存：慢但容量大
 *    - 常量内存：只读但有缓存加速
 *
 * 7. 程序输出解读：
 *    - 线程块0输出元素0.0到31.0
 *    - 线程块1输出元素32.0到63.0
 *    - 每行显示：数值、来源线程块索引
 */