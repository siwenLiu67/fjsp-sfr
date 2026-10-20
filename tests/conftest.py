from fjsp_sfr.core.models import (
    Checkpoint,
    FJSPInstance,
    Job,
    Machine,
    Operation,
    OperationAlternative,
    ServiceEntity,
)


def _make_job(
    job_id: int,
    *,
    due_date: int,
    quantity: int,
    first_processing: int,
    second_processing: int,
) -> Job:
    return Job(
        job_id=job_id,
        entity_id=0,
        release_time=0,
        due_date=due_date,
        quantity=quantity,
        operations=(
            Operation(
                op_id=job_id * 2,
                job_id=job_id,
                sequence_index=0,
                alternatives=(
                    OperationAlternative(0, first_processing),
                    OperationAlternative(1, first_processing + 1),
                ),
            ),
            Operation(
                op_id=job_id * 2 + 1,
                job_id=job_id,
                sequence_index=1,
                alternatives=(OperationAlternative(1, second_processing),),
            ),
        ),
    )


def make_test_instance() -> FJSPInstance:
    """Return a minimal two-machine instance with active dispatch candidates."""

    return FJSPInstance(
        jobs=(
            _make_job(
                0,
                due_date=5,
                quantity=2,
                first_processing=2,
                second_processing=2,
            ),
            _make_job(
                1,
                due_date=8,
                quantity=3,
                first_processing=3,
                second_processing=1,
            ),
            _make_job(
                2,
                due_date=12,
                quantity=1,
                first_processing=1,
                second_processing=3,
            ),
        ),
        entities=(ServiceEntity(0, (Checkpoint(10, 6.0, 1.0),)),),
        machines=(Machine(0), Machine(1)),
    )
