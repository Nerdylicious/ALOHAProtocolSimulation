#Purpose: A simulation of the ALOHA protocol
#Name: Irish Medina
#Date: October 20, 2014
from collections import deque
from scipy.stats import planck, poisson, randint
import numpy as np
import simpy
import random

NUM_STATIONS = 8
NUM_REPLICATIONS = 10
TRANSIENT_TIME = 25
TERMINATE_TIME = 10000
STEADY_STATE_TIME = TERMINATE_TIME - TRANSIENT_TIME

class Arrival:

    def __init__(self, name, time):
        self.name = name
        self.time = time

class Frame:

    def __init__(self, start, end, frame_time):

        self.start = start
        self.end = end
        self.frame_time = frame_time
        self.retry = False

    def __repr__(self):
        return "Frame: start=%d, end=%d, frame_time=%d" % (self.start, self.end, self.frame_time)

class Station(object):

    frames_in_transmit = {}

    def __init__(self, env, name, exponential_mean, poisson_mean):

        self.env = env
        self.name = name

        self.exponential_mean = exponential_mean
        self.poisson_mean = poisson_mean

        self.server = simpy.Resource(self.env, capacity=1)

        #keep track of arrivals
        self.arrivals = deque()
        self.initial_reset_completed = False

        #number of messages currently in system
        self.n = 0
        self.last_event = 0

        self.nt = 0
        self.st = 0
        self.T = 0

        self.num_retries = 0
        self.num_initial_transmits = 0
        self.mean_retries = 0

        self.busy_time = 0
        self.steady_state_time = 0
        self.U = 0

        self.env.process(self.arrive())

    #Purpose: Generate report for a single station
    def generate_report(self):

        #must subtract transient time from total time
        self.steady_state_time = self.env.now - TRANSIENT_TIME 

        #mean transmit time
        if self.nt > 0:
            self.T = float(self.st)/self.nt

        #mean number of retries
        if self.num_initial_transmits > 0:
            self.mean_retries = float(self.num_retries)/self.num_initial_transmits

        #channel utilization
        self.U = float(self.busy_time)/self.steady_state_time

        print "%s Mean transmit time=%f, Mean retries=%f, Utilization=%.2f%%" % (self.name, self.T, self.mean_retries, self.U * 100)

    #Purpose: Reset statistical counters that have been accumulated during the transient period 
    def reset_statistical_counters(self):
        
        self.nt = 0
        self.st = 0
        self.num_retries = 0
        self.num_initial_transmits = 0
        self.busy_time = 0

    #Purpose: Generate a frame time using an exponential distribution
    def generate_frame_time(self):
        is_nonzero_frame_time = False
        while not is_nonzero_frame_time:
            #planck gives us discrete exponential values
            R = planck.rvs(self.exponential_mean, size=1)
            if R[0] > 0:
                frame_time = R[0]
                is_nonzero_frame_time = True
        return frame_time

    #Purpose: Create a frame for transmit
    def create_frame(self, frame_time):
        start = self.env.now
        end = self.env.now + frame_time
        frame = Frame(start, end, frame_time)
        return frame

    #Purpose: Add a frame to frames currently in transmit
    def add_frame_in_transmit(self, frame, frame_id):
        Station.frames_in_transmit[frame_id] = frame

    #Purpose: Remove a frame from frames currently in transmit
    def remove_frame_in_transmit(self, frame_id):
        return Station.frames_in_transmit.pop(frame_id)

    #Purpose: Check for collisions by looking at the current frames in transmit
    def check_collision(self, frame, frame_id):

        has_collision = False
        for key in Station.frames_in_transmit.keys():
            if key != frame_id:
                #if there is a collision
                if (Station.frames_in_transmit[key].end > frame.start) and (Station.frames_in_transmit[key].start < frame.end):
                    has_collision = True

        #set all frames to retry
        if has_collision:
            for key in Station.frames_in_transmit.keys():
                Station.frames_in_transmit[key].retry = True

    #Purpose: Determines the back-off delay to wait if there is a collision
    def wait(self):
        mean = 0.0025
        R = planck.rvs(mean, size=1)
        retry_time = R[0]
        yield self.env.timeout(retry_time)

    #Purpose: Transmit the frame (until success)
    def transmit(self, name):

        self.num_initial_transmits = self.num_initial_transmits + 1
        success = False
        frame_time = self.generate_frame_time()

        while not success:
            frame = self.create_frame(frame_time)
            frame_id = (self.name, name)
            self.add_frame_in_transmit(frame, frame_id)
            
            transmit_time = self.env.now
            yield self.env.timeout(frame.frame_time)

            self.check_collision(frame, frame_id)
            self.remove_frame_in_transmit(frame_id)

            #if there was a collision 
            if frame.retry:
                self.num_retries = self.num_retries + 1
                yield env.process(self.wait())
            else:
                #if transmit was a success 
                self.busy_time = self.busy_time + (self.env.now - transmit_time)
                success = True

    #Purpose: Wait in the queue until the station has successfully transmitted the current frame
    def wait_for_service(self, name):
        arrival = Arrival(name, self.env.now)
        self.arrivals.append(arrival)

        self.last_event = self.env.now
        self.n = self.n + 1

        with self.server.request() as req:

            yield req

            arrival = self.arrivals.popleft()
            wait = self.env.now - arrival.time 
            transmit_time = self.env.now

            yield env.process(self.transmit(name))

            self.nt = self.nt + 1
            self.st = self.st + (self.env.now - transmit_time)

            self.last_event = self.env.now
            self.n = self.n - 1

            #if we're at or past the transient period then do initial data deletion (reset statistical counters)
            if not self.initial_reset_completed and self.env.now >= TRANSIENT_TIME:
                self.reset_statistical_counters()
                self.initial_reset_completed = True

    #Purpose: Generate the inter-arrival times of new messages using a Poisson distribution
    def arrive(self):
        i = 0
        while True:
            R = poisson.rvs(self.poisson_mean, size=1)
            inter_t = R[0]
            yield self.env.timeout(inter_t)
            w = self.wait_for_service('Frame %d' % i)
            env.process(w)
            i += 1

#Purpose: Generate a report for a single replication
def generate_report_single_replication(mean_transmit_times, mean_num_retries, channel_utilizations, stations):

    total_st = 0
    total_nt = 0
    total_retries = 0
    total_initial_transmits = 0
    total_busy_time = 0;

    for station in stations:
        total_st += station.st
        total_nt += station.nt
        total_retries += station.num_retries
        total_initial_transmits += station.num_initial_transmits
        total_busy_time += station.busy_time

    #if got no successes then we have a mean transmit time of 0
    if total_nt > 0:
        mean_t = float(total_st)/total_nt
        mean_transmit_times.append(mean_t)
    else:
        mean_t = 0
        mean_transmit_times.append(mean_t)

    mean_r = float(total_retries)/total_initial_transmits
    mean_num_retries.append(mean_r)

    #utilization is the percentage of the channel bandwidth that was used for transmitting frames without collisions
    #total_busy_time is the total time it took to transmit a frame successfully for all stations in the system
    #total_busy_time does not count the time it took to transmit a frame but collided (was not a success)
    mean_U = float(total_busy_time)/STEADY_STATE_TIME
    channel_utilizations.append(mean_U)

    print "Report for the whole system (all stations):"
    print "Mean transmit time=%f, Mean number retries=%f, Channel utilization=%.2f%%" % (mean_t, mean_r, mean_U * 100)

#Purpose: Generate a report over all replications
def generate_report_all_replications(mean_transmit_times, mean_num_retries, channel_utilizations):

    mean_t = np.mean(mean_transmit_times)
    mean_r = np.mean(mean_num_retries)
    mean_U = np.mean(channel_utilizations)

    print "-----------------------"
    print "Report over all replications: (means are over all replications)"
    print "Mean transmit time=%f, Mean number retries=%f, Channel utilization=%.2f%%" % (mean_t, mean_r, mean_U * 100)

if __name__=='__main__':

    mean_transmit_times = []
    mean_num_retries = []
    channel_utilizations = []

    for r in range(NUM_REPLICATIONS):
        print "-----------------------"
        print "Replication %d" % (r + 1)
        env = simpy.Environment()
        exponential_mean = 0.5
        #poisson_means = [1, 4, 10, 16, 20, 1, 4, 10]
        poisson_mean = 10
        stations = [Station(env, 'Station %d' % i, exponential_mean, poisson_mean) for i in range(NUM_STATIONS)]
        env.run(until=TERMINATE_TIME)
        print "Report for each Station:"
        for station in stations:
            station.generate_report()
        generate_report_single_replication(mean_transmit_times, mean_num_retries, channel_utilizations, stations)
    generate_report_all_replications(mean_transmit_times, mean_num_retries, channel_utilizations)
