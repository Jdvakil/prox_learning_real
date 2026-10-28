import pyrealsense2 as rs

pipeline = rs.pipeline()
pipeline.start()
frames = pipeline.wait_for_frames()
pipeline.stop()
