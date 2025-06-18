from prefect import task, flow


@task(cache_key_fn=lambda context, parameters: f"{parameters['argument']}")
def cachacble_task(argument: int):
    return argument


@flow
def the_flow(argument: int):
    cachacble_task(argument=argument)
    cachacble_task(argument=argument + 1)
    cachacble_task(argument=argument + 2)


if __name__ == "__main__":
    the_flow(argument=2)
