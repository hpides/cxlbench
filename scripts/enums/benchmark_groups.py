from enum import StrEnum


class BMGroups(StrEnum):
    RANDOM_WRITES = "random_writes"
    RANDOM_READS = "random_reads"
    SEQUENTIAL_WRITES = "sequential_writes"
    SEQUENTIAL_READS = "sequential_reads"
    OPERATION_LATENCY = "operation_latency"

    def get_title(self) -> object:
        if self == self.RANDOM_WRITES:
            return "Random Writes"
        elif self == self.RANDOM_READS:
            return "Random Reads"
        elif self == self.SEQUENTIAL_WRITES:
            return "Sequential Writes"
        elif self == self.SEQUENTIAL_READS:
            return "Sequential Reads"
        elif self == self.OPERATION_LATENCY:
            return "Operation Latency"
