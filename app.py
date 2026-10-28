"""面向入门开发者的响应式课程入口。控件与计算分属不同 cell。"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys
    import importlib
    _start = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    repo_root = next(p for p in (_start, *_start.parents) if (p / "GOAL.md").exists())
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import marimo as mo
    import chapters.learning as learning
    importlib.reload(learning)
    CHAPTERS = learning.CHAPTERS
    from chapters.chapter0_hardware import ui as lesson0
    from chapters.chapter1_bev import ui as lesson1
    from chapters.chapter2_baseline import ui as lesson2
    from chapters.chapter3_latent_vicreg import ui as lesson3
    from chapters.chapter4_joint_world_action import ui as lesson4
    from chapters.chapter5_stress_testing import ui as lesson5
    from chapters.chapter6_action_conditioned import ui as lesson6
    from chapters.chapter7_candidate_planning import ui as lesson7
    for _lesson in (lesson0, lesson1, lesson2, lesson3, lesson4, lesson5, lesson6, lesson7):
        importlib.reload(_lesson)
    return (
        CHAPTERS,
        lesson0,
        lesson1,
        lesson2,
        lesson3,
        lesson4,
        lesson5,
        lesson6,
        lesson7,
        mo,
    )


@app.cell
def _(CHAPTERS, mo):
    chapter_select = mo.ui.radio(options=CHAPTERS, value=next(iter(CHAPTERS)), label="选择学习章节")
    mo.sidebar([
        mo.md("# bev-wam-x\n\n从一张地图，学到世界模型。"),
        chapter_select,
        mo.md("**每次学一件事**\n\n选一个小节 → 动手操作 → 看画面与解释 → 展开中文代码。\n\n滑块可拖动，也可用左右方向键逐步查看。切换章节会保留已有参数。"),
    ])
    return (chapter_select,)


@app.cell
def _(lesson0):
    ch0_controls = lesson0.make_controls()
    return (ch0_controls,)


@app.cell
def _(lesson1):
    ch1_controls = lesson1.make_controls()
    return (ch1_controls,)


@app.cell
def _(lesson2):
    ch2_controls = lesson2.make_controls()
    return (ch2_controls,)


@app.cell
def _(lesson3):
    ch3_controls = lesson3.make_controls()
    return (ch3_controls,)


@app.cell
def _(lesson4):
    ch4_controls = lesson4.make_controls()
    return (ch4_controls,)


@app.cell
def _(lesson5):
    ch5_controls = lesson5.make_controls()
    return (ch5_controls,)


@app.cell
def _(
    ch0_controls,
    ch1_controls,
    ch2_controls,
    ch3_controls,
    ch4_controls,
    ch5_controls,
    ch6_controls,
    ch7_controls,
    chapter_select,
    lesson0,
    lesson1,
    lesson2,
    lesson3,
    lesson4,
    lesson5,
    lesson6,
    lesson7,
):
    selected = chapter_select.value
    _lesson, _controls = {
        "ch0": (lesson0, ch0_controls), "ch1": (lesson1, ch1_controls),
        "ch2": (lesson2, ch2_controls), "ch3": (lesson3, ch3_controls),
        "ch4": (lesson4, ch4_controls), "ch5": (lesson5, ch5_controls),
        "ch6": (lesson6, ch6_controls),
        "ch7": (lesson7, ch7_controls),
    }[selected]
    # 控件由上游cell拥有；此处读取value并显示相同控件，避免操作后重置。
    content = _lesson.render(_controls.value, _controls)
    content
    return


@app.cell(hide_code=True)
def _(lesson6):
    ch6_controls = lesson6.make_controls()
    return (ch6_controls,)


@app.cell(hide_code=True)
def _(lesson7):
    ch7_controls = lesson7.make_controls()
    return (ch7_controls,)


if __name__ == "__main__":
    app.run()
