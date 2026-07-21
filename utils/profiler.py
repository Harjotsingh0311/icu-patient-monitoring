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

        pipeline_avg = None

        for stage in sorted(self.total):

            avg = self.total[stage] / self.count[stage]

            print(f"{stage:<20}: {avg*1000:7.2f} ms")

            if stage == "Total Pipeline":
                pipeline_avg = avg

        print("--------------------------------------")

        if pipeline_avg is not None:
            print(f"{'Overall Time':<20}: {pipeline_avg*1000:7.2f} ms")
            print(f"{'Overall FPS':<20}: {1/pipeline_avg:7.2f}")

        print("======================================\n")

        self.total.clear()
        self.count.clear()