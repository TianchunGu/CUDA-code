/**
 * 课程 4.5: CUDA 动态共享内存详解
 * 文件名: dynamic_shared_memory.cu
 * 作者: 权 双
 * 日期: 2023-12-26
 * 功能: 动态共享内存使用演示
 *
 * 动态共享内存特点：
 * 1. 运行时分配 - 内核启动时指定大小
 * 2. 灵活性高 - 可以根据不同数据大小调整
 * 3. 外部声明 - 使用 extern __shared__ 声明
 * 4. 第三个启动参数 - kernel<<<grid, block, sharedMemSize>>>()
 * 5. 单一数组 - 如需多个数组，需要手动分区管理
 *
 * 动态 vs 静态共享内存对比：
 * - 静态：编译时确定，声明简单，类型安全
 * - 动态：运行时确定，更灵活，需要手动管理
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

/**
 * 动态共享内存声明
 *
 * extern __shared__ 语法说明：
 * - extern: 表示在其他地方定义，这里只是声明
 * - __shared__: 共享内存存储类别
 * - 数组大小: 在内核启动时通过第三个参数指定
 * - 类型: 这里使用float，但实际可以通过指针转换使用其他类型
 *
 * 重要注意事项：
 * - 只能声明一个动态共享内存数组
 * - 如需多个数组，需要手动分区这个单一数组
 * - 数组起始地址总是对齐到某个边界（通常是最大基本类型的大小）
 */
extern __shared__ float s_array[];

/**
 * GPU 内核函数 - 演示动态共享内存的使用
 *
 * @param d_A: 全局内存中的输入数组
 * @param N: 数组元素总数
 *
 * 与静态共享内存版本的主要区别：
 * 1. 共享内存大小在运行时确定
 * 2. 使用extern __shared__声明
 * 3. 内核启动时需要指定共享内存大小
 */
__global__ void kernel_1(float* d_A, const int N)
{
    /**
     * 获取线程标识符（与静态版本相同）
     */
    const int tid = threadIdx.x;  // 线程在块内的索引
    const int bid = blockIdx.x;   // 线程块索引
    const int n = bid * blockDim.x + tid;  // 全局线程索引

    /**
     * 第一阶段：协作加载数据到动态共享内存
     *
     * 访问模式分析：
     * - s_array现在是动态分配的
     * - 大小由内核启动时的第三个参数决定
     * - 访问方式与静态共享内存完全相同
     */
    if (n < N)
    {
        /**
         * 从全局内存加载到动态共享内存
         *
         * 关键观察：
         * - 语法与静态共享内存相同
         * - s_array[tid] 访问动态分配的共享内存
         * - 内存布局和访问模式保持一致
         */
        s_array[tid] = d_A[n];
    }

    /**
     * 线程同步（与静态版本相同）
     *
     * __syncthreads() 的必要性：
     * - 无论是静态还是动态共享内存，都需要同步
     * - 确保所有线程完成数据加载
     * - 避免后续访问时的数据竞争
     */
    __syncthreads();

    /**
     * 第二阶段：处理动态共享内存中的数据
     *
     * 处理逻辑与静态版本完全相同：
     * - 线程0负责输出所有数据
     * - 展示共享内存的共享特性
     * - 验证数据加载的正确性
     */
    if (tid == 0)
    {
        /**
         * 遍历动态共享内存数组
         *
         * 注意：这里硬编码了32个元素
         * 在实际应用中，可以通过参数传递实际大小
         * 或者使用blockDim.x作为循环上限
         */
        for (int i = 0; i < 32; ++i)
        {
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
        }
    }
}

/**
 * 主函数 - 演示动态共享内存的完整使用流程
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
     * 输出共享内存相关信息
     */
    std::cout << "每个线程块的共享内存大小: "
              << deviceProps.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "每个SM的共享内存大小: "
              << deviceProps.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;

    /**
     * 第二步：准备测试数据（与静态版本相同）
     */
    int nElems = 64;
    int nbytes = nElems * sizeof(float);

    /**
     * 分配主机内存并初始化
     */
    float* h_A = nullptr;
    h_A = (float*)malloc(nbytes);

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
     * 关键区别：动态共享内存的大小计算
     */
    dim3 block(32);
    dim3 grid(2);

    /**
     * 计算动态共享内存大小
     *
     * 计算逻辑：
     * - 每个线程块需要32个float元素
     * - 每个float占4字节
     * - 总共需要：32 × 4 = 128 字节
     *
     * 注意事项：
     * - 大小必须以字节为单位
     * - 系统会自动处理内存对齐
     * - 不能超过设备的共享内存限制
     */
    size_t sharedMemSize = 32 * sizeof(float);  // 128 字节

    std::cout << "\n内核配置:" << std::endl;
    std::cout << "线程块大小: " << block.x << std::endl;
    std::cout << "网格大小: " << grid.x << std::endl;
    std::cout << "动态共享内存大小: " << sharedMemSize << " 字节" << std::endl;

    /**
     * 第五步：启动内核函数 - 关键区别在这里
     *
     * 内核启动语法：kernel<<<grid, block, sharedMemSize>>>()
     *
     * 三个参数的含义：
     * - grid: 网格配置
     * - block: 线程块配置
     * - sharedMemSize: 动态共享内存大小（字节）
     *
     * 与静态版本的区别：
     * - 静态：kernel<<<grid, block>>>() - 2个参数
     * - 动态：kernel<<<grid, block, sharedMemSize>>>() - 3个参数
     */
    std::cout << "\n启动内核，输出结果:" << std::endl;
    kernel_1<<<grid, block, sharedMemSize>>>(d_A, nElems);

    /**
     * 等待内核完成
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
 * 1. 动态共享内存的声明和使用：
 *    - extern __shared__ 类型 数组名[];
 *    - 运行时通过内核启动参数指定大小
 *    - 第三个启动参数：<<<grid, block, sharedMemSize>>>
 *
 * 2. 与静态共享内存的对比：
 *    静态共享内存：
 *    - 优点：编译时检查，类型安全，声明简单
 *    - 缺点：大小固定，不够灵活
 *
 *    动态共享内存：
 *    - 优点：运行时确定大小，灵活性高
 *    - 缺点：需要手动管理，容易出错
 *
 * 3. 使用场景选择：
 *    - 静态：数据大小固定，追求安全性
 *    - 动态：数据大小变化，需要灵活配置
 *
 * 4. 多数组管理（动态共享内存的高级用法）：
 *    ```cuda
 *    extern __shared__ char s_data[];
 *    float* s_float = (float*)s_data;
 *    int* s_int = (int*)&s_float[float_count];
 *    ```
 *
 * 5. 性能考虑：
 *    - 动态和静态共享内存的访问速度相同
 *    - 主要区别在于管理便利性
 *    - 都需要考虑银行冲突优化
 *
 * 6. 常见错误：
 *    - 忘记在内核启动时指定大小
 *    - 大小计算错误（忘记乘以sizeof）
 *    - 多数组管理时的指针算术错误
 *    - 超出设备共享内存限制
 *
 * 7. 程序输出：
 *    - 与静态版本完全相同的输出
 *    - 证明了两种方法的功能等价性
 *    - 区别仅在于内存分配方式
 */