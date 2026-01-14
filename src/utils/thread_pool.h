#ifndef VOICE_ASSISTANT_THREAD_POOL_H
#define VOICE_ASSISTANT_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>

namespace voice_assistant {
namespace utils {

/**
 * @brief 线程池类
 *
 * 提供异步任务执行能力
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    // 禁止拷贝
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief 提交任务到线程池
     * @param func 可调用对象
     * @param args 参数
     * @return std::future 用于获取返回值
     */
    template<typename F, typename... Args>
    auto enqueue(F&& func, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

    /**
     * @brief 获取线程池大小
     */
    size_t size() const { return workers_.size(); }

    /**
     * @brief 获取等待中的任务数量
     */
    size_t queueSize() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

private:
    // 工作线程
    std::vector<std::thread> workers_;

    // 任务队列
    std::queue<std::function<void()>> tasks_;

    // 同步
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

// 模板实现
template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& func, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {

    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return res;
}

} // namespace utils
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_THREAD_POOL_H
