"""
Parent graph
  └ greet_customer
      └ child graph (choose_banchan)
          ├ pick_first_banchan
          └ add_to_plate

*서브그래프를 노드 함수 안에서 직접 호출
*부모/자식 state는 분리
*그래서 입출력 변환을 호출자가 직접 해야 함
"""

print(f"\n### invoke a graph from a node (반찬 고르기) ###\n")

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START


# ----------------------------
# Define subgraph (반찬 고르기 전용)
# ----------------------------
class BanchanSubState(TypedDict):
    base_plate: str  # 부모와 공유 X (부모가 변환해서 넣어줌)
    picked: str  # 서브그래프 내부에서 고른 반찬


def pick_first_banchan(state: BanchanSubState):
    # 예: 첫 반찬으로 김치 선택
    return {"picked": "김치"}


def add_to_plate(state: BanchanSubState):
    # 식판(base_plate)에 picked를 추가해서 새 식판 문자열을 만든다
    return {"base_plate": state["base_plate"] + " + " + state["picked"]}


sub_builder = StateGraph(BanchanSubState)
sub_builder.add_node(pick_first_banchan)
sub_builder.add_node(add_to_plate)
sub_builder.add_edge(START, "pick_first_banchan")
sub_builder.add_edge("pick_first_banchan", "add_to_plate")
choose_banchan_graph = sub_builder.compile()


# ----------------------------
# Define parent graph (손님 응대/전체 흐름)
# ----------------------------
class ParentState(TypedDict):
    plate: str  # 최종 식판


def greet_customer(state: ParentState):
    return {"plate": "식판 시작: " + state["plate"]}


def ask_and_add_banchan_via_invoke(state: ParentState):
    # 부모 -> 서브그래프로 상태 변환
    res = choose_banchan_graph.invoke({"base_plate": state["plate"]})
    # 서브그래프 -> 부모로 상태 변환
    return {"plate": res["base_plate"]}


builder = StateGraph(ParentState)
builder.add_node("greet_customer", greet_customer)
builder.add_node("add_banchan", ask_and_add_banchan_via_invoke)
builder.add_edge(START, "greet_customer")
builder.add_edge("greet_customer", "add_banchan")
graph = builder.compile()

print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :")
print(graph.get_graph(xray=True).draw_mermaid())

for chunk in graph.stream({"plate": "빈 식판"}, subgraphs=True):
    print(chunk)
"""
point: shared key

Parent graph (state: plate)
  ├ greet_customer
  └ choose_banchan (subgraph as a node)
       ├ pick_banchan
       └ update_plate

*부모 state 중 공유 키는 그대로
*서브그래프 private 키는 서브그래프 내부에서만 사용
*결과로 공유 키 업데이트만 부모로 전달
"""

print(f"\n### add a graph as a node (반찬 고르기) ###\n")

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START


# ----------------------------
# Define subgraph (부모와 plate를 공유)
# ----------------------------
class BanchanSubState(TypedDict):
    plate: str  # ✅ parent와 공유되는 키
    picked: str  # private (subgraph 내부용)


def pick_banchan(state: BanchanSubState):
    # 예: 이번엔 콩나물 선택
    return {"picked": "콩나물"}


def update_plate(state: BanchanSubState):
    # picked는 내부 키지만, 부모에게는 plate 업데이트만 전달됨
    return {"plate": state["plate"] + " + " + state["picked"]}


sub_builder = StateGraph(BanchanSubState)
sub_builder.add_node(pick_banchan)
sub_builder.add_node(update_plate)
sub_builder.add_edge(START, "pick_banchan")
sub_builder.add_edge("pick_banchan", "update_plate")
choose_banchan_graph = sub_builder.compile()


# ----------------------------
# Define parent graph
# ----------------------------
class ParentState(TypedDict):
    plate: str


def greet_customer(state: ParentState):
    return {"plate": "식판 시작: " + state["plate"]}


builder = StateGraph(ParentState)
builder.add_node("greet_customer", greet_customer)
builder.add_node("choose_banchan", choose_banchan_graph)  # ✅ subgraph를 노드로 추가
builder.add_edge(START, "greet_customer")
builder.add_edge("greet_customer", "choose_banchan")
graph = builder.compile()

print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :")
print(graph.get_graph(xray=True).draw_mermaid())

for chunk in graph.stream({"plate": "빈 식판"}, subgraphs=True):
    print(chunk)
print(f"\n### view subgraph state: only in interrupt (반찬 직접 선택) ###\n")

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict


class State(TypedDict):
    plate: str


# ----------------------------
# Subgraph: 사람에게 물어보고 반찬 추가
# ----------------------------
def ask_human_banchan(state: State):
    banchan = interrupt("어떤 반찬을 식판에 추가할까요? (예: 멸치볶음)")
    return {"plate": state["plate"] + " + " + banchan}


sub_builder = StateGraph(State)
sub_builder.add_node("ask_human_banchan", ask_human_banchan)
sub_builder.add_edge(START, "ask_human_banchan")
subgraph = sub_builder.compile()

# ----------------------------
# Parent graph: subgraph를 노드로 사용
# ----------------------------
builder = StateGraph(State)
builder.add_node("choose_banchan", subgraph)
builder.add_edge(START, "choose_banchan")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

# 1) 실행 -> interrupt 발생
graph.invoke({"plate": "빈 식판"}, config)

parent_state = graph.get_state(config)
print(f"[parent_state]: {parent_state}\n")

# interrupt 동안에만 subgraphs=True로 서브그래프 내부 snapshot 확인 가능
subgraph_task_state = graph.get_state(config, subgraphs=True).tasks[0].state
print(f"[subgraph_state during interrupt]: {subgraph_task_state}\n")

# 2) 사람 입력(resume)으로 재개
graph.invoke(Command(resume="멸치볶음"), config)

# 3) 실행이 끝나면 다시 서브그래프 내부는 잘 안 펼쳐짐(상황에 따라 tasks가 비거나 요약됨)
after = graph.get_state(config, subgraphs=True)
print(f"\n[after resume]: {after}")
