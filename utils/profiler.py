from collections import defaultdict


class PipelineProfiler:

    def __init__(self):

        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, stage: str, elapsed: float):

        self.total[stage] += elapsed
        self.count[stage] += 1

    def report(self):

        print("\n========== PIPELINE PROFILE ==========")

        overall = 0

        for stage in self.total:

            avg = self.total[stage] / self.count[stage]

            overall += avg

            print(f"{stage:<20}: {avg*1000:7.2f} ms")

        print("--------------------------------------")

        print(f"{'TOTAL':<20}: {overall*1000:7.2f} ms")

        if overall > 0:
            print(f"{'FPS':<20}: {1/overall:7.2f}")

        print("======================================\n")

        self.total.clear()
        self.count.clear()