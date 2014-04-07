from time import time

class TimeProfiler:
    
    def __init__(self):
        self.sections = {}
    
    def reset(self):
        self.sections = {}
    
    def start(self, section):

        if section not in self.sections.keys():
            self.sections[section] = 0
            
        self.curr_section = section
        self.tick = time()
        
    def stop(self):
        
        time_passed = (time() - self.tick) * 1000       # convert to ms
        self.sections[self.curr_section] = self.sections[self.curr_section] + time_passed
        
    def get_summary(self):
        
        """ Returns summary of each section showing percentage of total time and frames-per-second
        """
        
        total = sum(self.sections.values()) + 0.001     # hack to avoid division by 0
        summary = ''
        for section in self.sections.keys():
            summary = summary + '%s: %0.2f ' % (section, (self.sections[section] / float(total)) * 100)
            
        return summary + ('[total: %d ms, %d hz]' % (total, int(1000 / total)))
