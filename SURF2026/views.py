import json
from django.http import StreamingHttpResponse
from django.shortcuts import render
from .cahn import simulate_frames


COLORS = [
    "rgb(255,0,255)",
    "rgb(0,255,0)",
    "rgb(0,120,255)",
    "rgb(255,255,0)",
    "rgb(255,0,255)",
    "rgb(0,255,255)",
    "rgb(255,140,0)",
    "rgb(180,0,255)",
]


def index(request):
    t_max = float(request.GET.get("t_max", 5.0))
    step = float(request.GET.get("step", 0.1))
    kappa = float(request.GET.get("kappa", 5))
    N = int(request.GET.get("N", 3))

    context = {
        "t_max": t_max,
        "step": step,
        "kappa": kappa,
        "N": N,
        "colors": json.dumps(COLORS),
    }

    return render(request, "index.html", context)


def stream_simulation(request):
    t_max = float(request.GET.get("t_max", 5.0))
    step = float(request.GET.get("step", 0.1))
    kappa = float(request.GET.get("kappa", 5))
    N = int(request.GET.get("N", 3))

    def event_stream():
        for frame in simulate_frames(
            N=N,
            t_max=t_max,
            step=step,
            kappa=kappa,
            iterate=1,
        ):
            data = json.dumps(frame)
            yield f"data: {data}\n\n"

        yield "event: done\ndata: finished\n\n"

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )

    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"

    return response