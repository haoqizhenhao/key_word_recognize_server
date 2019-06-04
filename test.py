from threading import Thread
import queue, time

q = queue.Queue()


def consumer():
    while 1:
        res = q.get()
        time.sleep(2)
        print('消费者消费了%s' % res)
        q.task_done()


def producer_0():
    for i in range(5):
        q.put(i)
        print('生产者0生产了%s' % i)
    q.join()


def producer_1():
    for i in range(5):
        q.put(i)
        print('生产者1生产了%s' % i)
    q.join()


def producer_2():
    for i in range(5):
        q.put(i)
        print('生产者2生产了%s' % i)
    q.join()


if __name__ == '__main__':
    t0 = Thread(target=producer_0, )
    t1 = Thread(target=producer_1, )
    t2 = Thread(target=producer_2, )

    t0.start()
    t1.start()
    t2.start()
    consumer_t = Thread(target=consumer, )
    consumer_t.daemon = True
    consumer_t.start()
    t0.join()
    t1.join()
    t2.join()
    print('主线程~')