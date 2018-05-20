# coding: utf-8

def read_graph(filename):
    graph = {}
    with open(filename, 'rb') as f:
        for line in f:
            info = line.split(',')
            key = info[0]
            key = key.strip()
            graph[key] = []
            if key != 'node15':
                for ele in info[1:]:
                    node, cost = ele.split()
                    node = node.strip()
                    cost = cost.strip()
                    cost = float(cost)
                    graph[key].append((node, cost))
    return graph


def updateJ(graph, J):
    newJ = {}

    for key in graph:
        if key == 'node15':
            newJ['node15'] = 0.0
        else:
            nodes = graph[key]
            newJ[key] = min(node[1] + J[node[0]] for node in nodes)
    return newJ


def printPath(graph, J):
    start = 'node0'
    sum_cost = 0.0
    while start != 'node15':
        print(start)
        running_min = 1e4
        for dest, cost in graph[start]:
            cost_of_path = cost + J[dest]
            if cost_of_path < running_min:
                running_min = cost_of_path
                min_cost = cost
                min_dest = dest
        start = min_dest
        sum_cost += min_cost

    print('node99\n')
    print('total cost is {0:.2f}'.format(sum_cost))


if __name__ == '__main__':
    filename = r'C:\python-tutorial\graph.txt'
    J = {}
    graph = read_graph(filename)
    bigNum = 1e4
    for key in graph:
        J[key] = bigNum
    J['node15'] = 0.0

    iterTimes = 0
    while True:
        newJ = updateJ(graph, J)
        iterTimes += 1
        if newJ == J:
            break
        else:
            J = newJ

    printPath(graph, J)