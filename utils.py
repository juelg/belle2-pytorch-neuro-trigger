import logging
import threading

# TODO make threadsave
class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """

   def __init__(self, experts, level):
      self.loggers = {expert: logging.getLogger(expert) for expert in experts}

      self.level = level
      self.linebuf = ''

   def write(self, buf):
      f = self.loggers[threading.Thread.getName(threading.current_thread())]
      for line in buf.rstrip().splitlines():
         # self.loggers[threading.Thread.getName(threading.current_thread())].log(self.level, line.rstrip())
         f.log(self.level, line.rstrip())

   def flush(self):
      pass

class StreamToLogger2(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """

   def __init__(self, logger, level):
      self.logger = logger

      self.level = level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.level, line.rstrip())

   def flush(self):
      pass


class StreamToLogger3(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''