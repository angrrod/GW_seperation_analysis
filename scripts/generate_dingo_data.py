from Pipeline import DINGO_pipeline
from setUpLoggerScenario import setUpLoggerScenario
import os

def Main():
    print('test')
    model_name = "best_model.pt"
    scenario, logger, _, _ = setUpLoggerScenario()

    # If setUpLoggerScenario does not already call this, uncomment:
    # scenario.setUpScenario()

    DP = DINGO_pipeline(logger, scenario, generateData = True,nbr_cores=30)

    # ----------------------------------------------------
    # Optional node-local dataset cache.
    # This changes only our wrapper settings, not Dingo.
    # ----------------------------------------------------

    use_local_cache = (
        os.environ.get(
            "GW_USE_LOCAL_CACHE",
            "0",
        )
        == "1"
    )

    if use_local_cache:

        cache_dir = os.environ.get(
            "GW_LOCAL_CACHE"
        )

        if cache_dir is None:
            raise RuntimeError(
                "GW_USE_LOCAL_CACHE=1 but "
                "GW_LOCAL_CACHE is not defined."
            )

        DP.local_settings[
            "local_cache_path"
        ] = cache_dir

    else:

        DP.local_settings.pop(
            "local_cache_path",
            None,
        )

    print(
        "USE LOCAL CACHE:",
        use_local_cache,
    )

    print(
        "local_cache_path:",
        DP.local_settings.get(
            "local_cache_path"
        ),
    )

if __name__ == "__main__":
    Main()
    print("End")