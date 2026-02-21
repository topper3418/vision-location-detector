
"""Main entry point for vision location detector application.

Initializes video feed, detector postprocessor, and web server, then starts the application.
"""


import sys
import logging
from src.server_builder import ServerBuilder
from src.settings import settings


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Application settings: {settings}")

    builder = ServerBuilder()
    # build the default server
    builder.initialize()
    server, _ = builder.build()

    if server is None:
        logger.error("Failed to build server. Exiting.")
        return 1

    if server.video_feed is None:
        logger.error("Server has no video feed. Exiting.")
        return 1

    logger.info(f"Starting web server on {settings.server_host}:{settings.server_port}...")
    try:
        server.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt.")
        return 0
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return 1
    finally:
        if server.video_feed is not None:
            server.video_feed.release()


if __name__ == '__main__':
    sys.exit(main())
