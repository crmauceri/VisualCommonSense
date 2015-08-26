# Author: Edison Huang
# Email: hom.tao@yahoo.com
#
# Example:
# def main():      
#   from CmdArgumentsHelper import CmdArgumentsHelper;
#   arg_helper = CmdArgumentsHelper();
#   arg_helper.add_argument('query', 'q', 'query', 1);
#   args = arg_helper.read_arguments();
#
#   query_string = args['query'];
#   ... manipulating query_string ...
#
  
# if __name__ == "__main__":
#   main();
#

class CmdArgumentsHelper(object):
  args = [];
  args_cmd = {};
  args_option = {};
  args_has_value = {};

  def add_argument(self, argument_name, argument_cmd, argument_option, has_value):
    self.args.append(argument_name);
    self.args_cmd[argument_name] = argument_cmd;
    self.args_option[argument_name] = argument_option;
    self.args_has_value[argument_name] = has_value;

  def gen_help_message(self):
    help_message = '';
    for arg in self.args:
      help_message = help_message + ' -' + self.args_cmd[arg] + ' ' + '<' + arg + '>';
    return help_message;

  def gen_cmds(self):
    cmds = 'h';
    for arg in self.args:
      cmds = cmds + self.args_cmd[arg];
      if (self.args_has_value[arg]):
        cmds = cmds + ':';
    return cmds;

  def gen_options(self):
    options = [];
    for arg in self.args:
      if (self.args_has_value[arg]):
        options.append(self.args_option[arg] + '=');
    return options;
    
  def _read_arguments(self, argv):
    import sys, getopt;
    help_message = self.gen_help_message();
    try:
      opts, args = getopt.getopt(argv, self.gen_cmds(), self.gen_options());
    except:
      print (help_message);
      sys.exit(2);

    ret = {};
    for opt, arg_value in opts:
      for arg_name in self.args:
        if (opt in ('-' + self.args_cmd[arg_name], '--' + self.args_option[arg_name])):
          ret[arg_name] = arg_value;

    return ret;
  
  def read_arguments(self):
    import sys;
    return self._read_arguments(sys.argv[1:]);
    
