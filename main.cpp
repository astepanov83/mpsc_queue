#include <atomic>
#include <cstdint>
#include <iostream>
#include <new>
#include <thread>
#include <vector>

using Integer = int64_t;

class IEvent
{
public:
  virtual ~IEvent() = default;
  virtual void Process() = 0;
};

template <class T>
struct alignas(std::hardware_destructive_interference_size) PaddedType
{
  T value_;
  uint8_t padding_[std::hardware_destructive_interference_size - sizeof(value_)];

  PaddedType() = default;

  PaddedType(T t)
    : value_{std::move(t)}
  {
  }
};

using PaddedInteger = PaddedType<Integer>;
using PaddedAtomicInteger = PaddedType<std::atomic<Integer>>;

static_assert(sizeof(PaddedInteger) == std::hardware_destructive_interference_size);
static_assert(sizeof(PaddedAtomicInteger) == std::hardware_destructive_interference_size);

template <class E>
class EventProcessor
{
public:
  EventProcessor(size_t queue_capacity_exp = 16)
    : queue_capacity_exp_{queue_capacity_exp}
    , queue_capacity_{static_cast<size_t>(1 << queue_capacity_exp)}
    , queue_(queue_capacity_)
    , commits_(queue_capacity_)
    , reserve_sn_{0}
    , commit_sn_{0}
    , consume_sn_{0}
  {
    // if it's zero, then this element will be auto-committed on the first Commit execution even if it wasn't really committed
    commits_[0].value_ = -1;
  }

  bool Reserve(size_t count, Integer& sequence_number)
  {
    // can't reserve more than a queue size
    if (count > queue_capacity_)
      return false;

    Integer reserve_sn;
    Integer consume_sn;
    do
    {
      reserve_sn = reserve_sn_.load(std::memory_order_relaxed);
      consume_sn = consume_sn_.load(std::memory_order_relaxed);

      // can't reserve more than the number offree items in the queue
      if (reserve_sn - consume_sn > queue_capacity_ - count)
        return false;
    }
    while (!reserve_sn_.compare_exchange_weak(reserve_sn, reserve_sn + count, std::memory_order_acquire));

    sequence_number = reserve_sn;

    return true;
  }

  E& ElementAt(Integer sequence_number)
  {
    return queue_[Index(sequence_number)].value_;
  }

  void Commit(size_t count, Integer sequence_number)
  {
    // mark elements as committed
    for (size_t i{}; i < count; ++i)
      commits_[Index(i + sequence_number)].value_.store(i + sequence_number, std::memory_order_release);

    auto reserve_sn = reserve_sn_.load(std::memory_order_relaxed);
    auto commit_sn = commit_sn_.load(std::memory_order_relaxed);

    // move commit_sn_ forward if possible
    while (commit_sn < reserve_sn)
    {
      if (!commits_[Index(commit_sn)].value_.compare_exchange_strong(commit_sn, -1, std::memory_order_relaxed))
        break;
      commit_sn_.store(++commit_sn, std::memory_order_release);
    }
  }

  bool Consume(E& event)
  {
    Integer commit_sn;
    Integer consume_sn;
    do
    {
      commit_sn = commit_sn_.load(std::memory_order_acquire);
      consume_sn = consume_sn_.load(std::memory_order_relaxed);

      if (commit_sn == consume_sn)
        return false;

      // save event before moving consume_sn_ otherwise it may be overwritten if the queue is full
      event = queue_[Index(consume_sn)].value_;
    }
    while (!consume_sn_.compare_exchange_weak(consume_sn, consume_sn + 1, std::memory_order_acquire));

    return true;
  }

private:
  size_t Index(Integer sn)
  {
    return static_cast<size_t>(sn) & ((1 << queue_capacity_exp_) - 1);
  }

private:
  size_t queue_capacity_exp_;
  size_t queue_capacity_;
  std::vector<PaddedType<E>> queue_;
  std::vector<PaddedAtomicInteger> commits_;
  alignas(std::hardware_destructive_interference_size) std::atomic<Integer> reserve_sn_;
  alignas(std::hardware_destructive_interference_size) std::atomic<Integer> commit_sn_;
  alignas(std::hardware_destructive_interference_size) std::atomic<Integer> consume_sn_;
};

class Event : public IEvent
{
public:
  Event(Integer sn)
    : sn_{sn}
  {
  }

  virtual void Process()
  {
    // std::cout << "Processed " + std::to_string(sn_) + "\n";
  }

private:
  Integer sn_;
};

int main()
{
  EventProcessor<IEvent*> ep{4};

  std::jthread consumer{[&]() {
    size_t totalEvents{};
    for (;;)
    {
      IEvent* event;
      while (!ep.Consume(event))
        std::this_thread::yield();

      // let this be the stop condition
      if (!event)
      {
        std::cout << "Total events processed: " << totalEvents << std::endl;
        break;
      }

      event->Process();
      delete event;
      ++totalEvents;
    }
  }};

  std::vector<std::jthread> producers;
  for (size_t i{}; i < 16; ++i)
  {
    producers.emplace_back([&, i](){
      auto reserve_count = i % 5 + 1;
      Integer sequence_number{};
      do
      {
        while (!ep.Reserve(reserve_count, sequence_number))
          std::this_thread::yield();

        for (size_t ri{}; ri < reserve_count; ++ri)
        {
          // std::cout << "Reserved " + std::to_string(sequence_number + ri) + " at thread " + std::to_string(i) + "\n";
          ep.ElementAt(sequence_number + ri) = new Event(sequence_number + ri);
        }

        ep.Commit(reserve_count, sequence_number);
      }
      while (sequence_number < 1048576);
    });
  }

  // wait till all the producers are done
  producers.clear();

  // push null event which is a stop condition
  Integer sequence_number;
  while (!ep.Reserve(1, sequence_number))
    std::this_thread::yield();

  ep.ElementAt(sequence_number) = nullptr;
  ep.Commit(1, sequence_number);
}
