"""Auto-generated code from UML model"""

class OpenDoor:
    pass

class DoorClosed:
    pass

class DoorOpening:
    pass

class DoorOpen:
    pass

    def activateMotor(self):
        start_time = time.time()
        # TODO: Implement operation
        pass
        duration = time.time() - start_time
        if duration < 0.0:
            warnings.warn(f"Operation {{self.name}} completed too fast: {{duration}}s")
        if duration > 5000.0:
            warnings.warn(f"Operation {{self.name}} exceeded max duration: {{duration}}s")

    def checkSafety(self):
        start_time = time.time()
        # TODO: Implement operation
        pass
        duration = time.time() - start_time
        if duration < 0.0:
            warnings.warn(f"Operation {{self.name}} completed too fast: {{duration}}s")
        if duration > 5000.0:
            warnings.warn(f"Operation {{self.name}} exceeded max duration: {{duration}}s")

class CloseDoor:
    pass

    def detectObstacle(self):
        start_time = time.time()
        # TODO: Implement operation
        pass
        duration = time.time() - start_time
        if duration < 0.0:
            warnings.warn(f"Operation {{self.name}} completed too fast: {{duration}}s")
        if duration > 5000.0:
            warnings.warn(f"Operation {{self.name}} exceeded max duration: {{duration}}s")
