from threading import BoundedSemaphore
emotion = None
face_coords = None
debug = True
semaphor = BoundedSemaphore(value=1)