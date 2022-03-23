# 알고리즘

<br><br><br>

# 그래프 탐색 알고리즘

## 📝 DFS(Depth First Search)

**DFS(Depth First Search)** 는 `깊이 우선 탐색`이라고도 부르며, 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘이다.

일반적으로 DFS는 `스택으로 구현`하며, 재귀를 이용하면 좀 더 간단하게 구현할 수 있다.

<details>
<summary>Code</summary>
<div>

```python
# 2차원 배열로 입력 받을 때
def dfs(graph, v, visited):
    visited[v] = True
    print(v, end=' ')
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)
```

```python
# 딕셔너리로 입력 받을 때
def dfs(graph, v, visited=[]):
    visited.append(v)
    for node in graph[v]:
        if node not in visited:
            dfs(graph, node, visited)

    return visited
```

</div>
</details>

<br><br><br>

## 📝 BFS(Breadth First Search)

**BFS(Breadth First Search)** 알고리즘은 `너비 우선 탐색`이라는 의미를 가진다. DFS는 최대한 멀리 있는 노드를 우선으로 탐색하는 방식으로 동작하지만, BFS는 `가까운 노드부터 탐색`하는 알고리즘이다.<br>
BFS는 `큐를 사용하여 구현`한다.

<details>
<summary>Code</summary>
<div>

```python
# 2차원 배열로 입력 받을 때
from collections import deque

def bfs(graph, start, visited):
    visited[start] = True
    queue = deque([start])

    while queue:
        v = queue.popleft()
        print(v, end=' ')
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
```

```python
# 딕셔너리로 입력 받을 때
from collections import deque

def bfs(graph, start):
    visited = []
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            queue.extend(graph[node])

    return visited
```

</div>
</details>

<br><br><br>

# 정렬 알고리즘

## 📝 버블 정렬(Bubble Sort)

두 인접한 데이터를 비교해서 앞에 있는 데이터가 뒤에 있는 데이터 보다 크면 자리를 바꾸는 정렬 알고리즘이다.

<br>

### 버블 정렬의 시간 복잡도

코드가 직관적이고, 구현하기 쉽지만, 최선이든 최악이든 $O(n^2)$이라는 시간복잡도를 가진다. 원소의 개수가 많아지면 비교 횟수가 많아져 **성능이 저하**된다.

|   최악   |   평균   |   최선   |
| :------: | :------: | :------: |
| $O(n^2)$ | $O(n^2)$ | $O(n^2)$ |

<details>
<summary>Code</summary>
<div>

```python
def bubble_sort(data):
    for i in range(len(data) - 1, 0, -1):
        for j in range(i):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data
```

</div>
</details>

<br><br><br>

## 📝 선택 정렬(Selection Sort)

데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 교체하고, 그다음 작은 데이터를 선택해 앞에서 두번째 데이터와 바꾸는 정렬이다.

**매번 가장 작은 것을 선택한다**는 의미에서 **선택 정렬(Selection Sort)** 이라 한다.

<br>

### 선택 정렬의 시간 복잡도

| 최악     | 평균     | 최선     |
| -------- | -------- | -------- |
| $O(N^2)$ | $O(N^2)$ | $O(N^2)$ |

선택 정렬의 시간 복잡도는 $O(N^2)$이다. 소스코드 상으로 간단한 형태의 2중 반복문이 사용되었기 때문이다.

<details>
<summary>Code</summary>
<div>

```python
def selection_sort(data):
    for i in range(len(data)):
        min_index = i
        for j in range(i + 1, len(data)):
            if data[min_index] > data[j]:
                min_index = j
        data[min_index], data[i] = data[i], data[min_index]
    return data
```

</div>
</details>

<br><br><br>

## 📝 삽입 정렬(Insertion Sort)

삽입 정렬은 특정한 데이터를 적절한 위치에 삽입한다는 의미에서 **삽입 정렬(Insertion Sort)** 이라고 한다.

삽입 정렬은 필요할 때만 위치를 바꾸므로 **데이터가 거의 정렬되어 있을 때** 훨씬 효율적이다.

<br>

### 삽입 정렬의 시간 복잡도

| 최악     | 평균     | 최선   |
| -------- | -------- | ------ |
| $O(N^2)$ | $O(N^2)$ | $O(N)$ |

<details>
<summary>Code</summary>
<div>

```python
def insertion_sort(data):
    for i in range(len(data)):
        for j in range(i, 0, -1):
            if data[j] < data[j - 1]:
                data[j], data[j - 1] = data[j - 1], data[j]
            else:
                break
    return data
```

</div>
</details>

<br><br><br>

## 📝 퀵 정렬(Quick Sort)

기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸면서 정렬하는 알고리즘이다.

퀵 정렬에는 **`피벗(pivot)`** 이 사용된다.

<br>

### 퀵 정렬의 시간 복잡도

| 최악     | 평균       | 최선       |
| -------- | ---------- | ---------- |
| $O(N^2)$ | $O(NlogN)$ | $O(NlogN)$ |

<details>
<summary>Code</summary>
<div>

```python
def quick_sort(data, start, end):
    if start >= end:
        return
    pivot = start
    left = start + 1
    right = end

    while left <= right:
        while left <= end and data[left] <= data[pivot]:
            left += 1
        while right > start and data[right] >= data[pivot]:
            right -= 1

        if left > right:
            data[right], data[pivot] = data[pivot], data[right]
        else:
            data[left], data[right] = data[right], data[left]

    quick_sort(data, start, right - 1)
    quick_sort(data, right + 1, end)

    return data
```

### pythonic한 코드

피벗과 데이터를 비교하는 비교 연산 횟수가 증가하므로 시간 면에서는 조금 비효율적이다.

```python
def quick_sort(data):
    if len(data) <= 1:
        return data

    pivot = data[0]

    left = [item for item in data[1:] if pivot > item]
    right = [item for item in data[1:] if pivot < item]

    return quick_sort(left) + [pivot] + quick_sort(right)
```

</div>
</details>

<br><br><br>

## 📝 계수 정렬(Counting Sort)

계수 정렬은 특정한 조건이 부합할 때만 사용할 수 있는 매우 빠른 알고리즘이다.

계수 정렬은 **데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때**만 사용할 수 있다.

리스트를 선언하고 그안에 데이터를 넣고 데이터를 하나씩 확인하며 데이터의 값과 동일한 인덱스의 데이터를 1씩 증가시키는 방식이다.

<br>

### 계수 정렬의 시간 복잡도

데이터의 개수를 $N$, 데이터 중 최대값의 크기를 $K$라고 할때, 계수 정렬의 시간 복잡도는 $O(N+K)$이다.

<br>

### 계수 정렬의 공간 복잡도

계수 정렬은 데이터의 크기가 한정되어 있고, 데이터의 크기가 많이 중복되어 있을수록 유리하며 항상 사용할 수는 없다.

계수 정렬은 동일한 값을 가지는 데이터가 여러 개 등장할 때 적합하다. 데이터의 특성을 파악하기 어렵다면 퀵 정렬을 이용하는 것이 유리하다.

<details>
<summary>Code</summary>
<div>

```python
def count_sort(data):
    count = [0] * (max(data) + 1)
    result = []

    for i in range(len(data)):
        count[data[i]] += 1

    for i in range(len(count)):
        for j in range(count[i]):
            result.append(i)

    return result
```

</div>
</details>

<br><br><br>
