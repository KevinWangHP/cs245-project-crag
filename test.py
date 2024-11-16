from collections import defaultdict


def execute_task(task, dependencies, skip_tasks):
    if task in skip_tasks:
        return []
    res = [task]
    if task in dependencies:
        for sub_task in dependencies[task]:
            res.extend(execute_task(sub_task, dependencies, skip_tasks))
    return res


def task_c():

    tasks_order = input().strip().split()

    N = int(input().strip()[0])

    dependencies = defaultdict(list)
    for i in range(N):
        parts = input().strip().split()
        task = parts[0]
        sub_tasks = parts[1:]
        dependencies[task].extend(sub_tasks)

    specified_tasks = set(input().strip().split())
    skip_tasks = set(input().strip().split())

    original_order = []
    for i in range(len(tasks_order)):
        original_order.append(execute_task(tasks_order[i], dependencies, skip_tasks))

    specified_res = []
    if specified_tasks:
        for i in original_order:
            curset = set(i)
            if not curset.isdisjoint(specified_tasks):
                specified_res.append(i)
    else:
        for i in original_order:
            specified_res.append(i)

    res = []
    for i in specified_res:
        res.extend(i)

    print(" ".join(res))


task_c()
