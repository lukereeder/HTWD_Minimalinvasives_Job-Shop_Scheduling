from decimal import Decimal
from typing import Optional

from config.project_config import get_solver_logs_path
from src.EmailNotifier import EmailNotifier
from src.Logger import Logger
from src.domain.Collection import LiveJobCollection
from src.domain.Query import JobQuery, ExperimentQuery, MachineQuery, MachineInstanceQuery
from src.domain.orm_models import Experiment
from src.simulation.LognormalFactorGenerator import LognormalFactorGenerator
from src.simulation.ProductionSimulation import ProductionSimulation
from src.solvers.CP_Solver import Solver

# email_notifier = EmailNotifier()  # Deaktiviert für lokale Läufe
email_notifier = None

def run_experiment(
        experiment_id: int,  shift_length: int, total_shift_number: int, logger: Logger,
        time_limit: Optional[int] = 60*20, bound_warmup_time: int = 30, bound_no_improvement_time: Optional[int] = 60,
        use_time_weighted_deviation: bool = False,
        deviation_window_minutes: int = 8 * 60,
        deviation_bucket_minutes: int = 60,
        deviation_max_factor: Optional[int] = None,
        machine_blockades: Optional[list[dict]] = None,
):
    experiment = ExperimentQuery.get_experiment(experiment_id)

    source_name = experiment.routing_source.name
    max_bottleneck_utilization = experiment.max_bottleneck_utilization

    w_t, w_e, w_dev = experiment.get_solver_weights()

    # Preparation  ----------------------------------------------------------------------------------
    simulation = ProductionSimulation(verbose=False)

    # Jobs Collection
    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        arrival_limit=60 * 24 * total_shift_number
    )
    jobs_collection = LiveJobCollection(jobs)

    # Machines with transition times
    machines_instances = MachineInstanceQuery.get_by_source_name_and_max_bottleneck_utilization(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
    )

    # Add transition times to operations
    for machine_instance in machines_instances:
        for job in jobs_collection.values():
            for operation in job.operations:
                if operation.machine_name == machine_instance.name:
                    operation.transition_time = machine_instance.transition_time

    # Add simulation durations to operations
    factor_gen = LognormalFactorGenerator(
        sigma=experiment.sim_sigma,
        seed=42
    )
    jobs_collection.sort_jobs_by_id()
    jobs_collection.sort_operations()
    for job in jobs_collection.values():
        for operation in job.operations:
            sim_duration_float = operation.duration * factor_gen.sample()
            operation.sim_duration = int(sim_duration_float)


    # Collections(empty)
    schedule_jobs_collection = LiveJobCollection()  # pseudo previous schedule
    active_job_ops_collection = LiveJobCollection()

    waiting_job_ops_collection = LiveJobCollection()

    shift_summaries: list[dict] = []

    # Shifts ----------------------------------------------------------------------------------------
    for shift_number in range(1, total_shift_number + 1):
        shift_start = shift_number * shift_length
        shift_end = (shift_number + 1) * shift_length
        logger.info(f"Experiment {experiment_id} shift {shift_number}: {shift_start} to {shift_end}")

        new_jobs_collection = jobs_collection.get_subset_by_earliest_start(earliest_start=shift_start)
        current_jobs_collection = new_jobs_collection + waiting_job_ops_collection

        active_blockades = []
        if machine_blockades:
            for blockade in machine_blockades:
                if blockade['start'] < shift_end and blockade['end'] > shift_start:
                    active_blockades.append(blockade)
                    logger.info(f"Active blockade in shift {shift_number}: {blockade['machine']} from {blockade['start']} to {blockade['end']}")

        # Scheduling --------------------------------------------------------------
        solver = Solver(
            jobs_collection=current_jobs_collection,
            logger = logger,
            schedule_start=shift_start,
            machine_blockades=active_blockades
        )

        if use_time_weighted_deviation:
            solver.build_model__absolute_lateness__time_weighted_start_deviation__minimization(
                previous_schedule_jobs_collection=schedule_jobs_collection,
                active_jobs_collection=active_job_ops_collection,
                w_t=w_t, w_e=w_e, w_dev=w_dev,
                deviation_window_minutes=deviation_window_minutes,
                deviation_bucket_minutes=deviation_bucket_minutes,
                deviation_max_factor=deviation_max_factor,
            )
        else:
            solver.build_model__absolute_lateness__start_deviation__minimization(
                previous_schedule_jobs_collection=schedule_jobs_collection,
                active_jobs_collection=active_job_ops_collection,
                w_t=w_t, w_e=w_e, w_dev=w_dev
            )

        solver.log_model_info()

        file_path = get_solver_logs_path(
            sub_directory=f"Experiment_{experiment_id:03d}",
            file_name=f"Shift_{shift_number:02d}.log",
            as_string=True
        )

        solver.solve_model(
            gap_limit=0.002,
            time_limit=time_limit,
            log_file=file_path,
            bound_relative_change= 0.01,
            bound_no_improvement_time= bound_no_improvement_time,
            bound_warmup_time=bound_warmup_time,
        )

        solver.log_solver_info()
        shift_summaries.append({"shift": shift_number, **solver.get_solver_info()})
        schedule_jobs_collection = solver.get_schedule()

        ExperimentQuery.save_schedule_jobs(
            experiment_id=experiment_id,
            shift_number=shift_number,
            live_jobs=schedule_jobs_collection.values(),
        )

        # Simulation --------------------------------------------------------------
        simulation.run(
            schedule_collection=schedule_jobs_collection,
            start_time=shift_start,
            end_time=shift_end
        )

        active_job_ops_collection = simulation.get_active_operation_collection()
        waiting_job_ops_collection = simulation.get_waiting_operation_collection()

        if shift_number == 1:
            notify(experiment, logger, shift_number, last_lines=30)
        elif shift_number % 10 == 0:
            notify(experiment, logger, shift_number, last_lines=100)

    # Save entire Simulation -------------------------------------------------------
    entire_simulation_jobs = simulation.get_entire_finished_operation_collection()
    ExperimentQuery.save_simulation_jobs(
        experiment_id=experiment_id,
        live_jobs=entire_simulation_jobs.values(),
    )
    logger.info(f"Experiment {experiment_id} finished")
    notify(experiment, logger, last_lines= 2)
    return {
        "experiment_id": experiment_id,
        "use_time_weighted_deviation": use_time_weighted_deviation,
        "machine_blockades": machine_blockades,
        "shift_summaries": shift_summaries,
    }


def notify(experiment:Experiment, logger: Logger, shift_number: Optional[int] = None, last_lines: int = 10):
    if email_notifier is None:
        return  # Email-Benachrichtigung deaktiviert
    
    experiment_info = f"Experiment {experiment.id} "
    if shift_number:
        experiment_info += (f"Shift {shift_number} - "
                            + f"Absolute Lateness ratio: {experiment.absolute_lateness_ratio}, "
                            + f"Inner Tardiness ratio: {experiment.inner_tardiness_ratio}, "
                            + f"Max bottleneck utilization: {experiment.max_bottleneck_utilization}, "
                            + f"Simulation sigma: {experiment.sim_sigma}")
    else:
        experiment_info += "finished"

    email_notifier.send_log_tail(
        subject=f"{experiment_info}",
        log_file= logger.get_log_file_path(),
        lines = last_lines
    )




