# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Prefix Postfix Utilities"""
import json
import pickle

from hypothesis_generation.hypothesis_utils import Node
from hypothesis_generation.hypothesis_utils import GrammarExpander
from hypothesis_generation.execute_hypotheses_on_scene_json import FunctionBinders


class PrefixPostfix(object):

  def __init__(self, properties_file, grammar_expander_file=None, grammar_expander=None):
    if grammar_expander_file is None and grammar_expander is None:
      raise ValueError("Expect to be provided either expander serialized or expander object.")
    
    with open(properties_file, 'r') as f:
      properties_metadata = json.load(f)
      metadata = properties_metadata["metadata"]
      size_mapping = properties_metadata["properties"]["sizes"]

    if grammar_expander is None:
      with open(grammar_expander_file, 'rb') as f:
        grammar_expander = pickle.load(f)
    constants_in_grammar = grammar_expander.terminal_constants

    function_binders = FunctionBinders(metadata, size_mapping,
                                       constants_in_grammar)
    self._function_binders = function_binders

  def postfix_to_tree(self, postfix_program):
    execution_stack = []
    if not isinstance(postfix_program, str):
      raise ValueError

    postfix_program = [x for x in postfix_program.split(' ') if x != ' ']

    quantifier_reached = 0
    for token in postfix_program:
      if token == 'lambda' or token == 'S.':
        quantifier_reached = 1
        continue

      if token in self._function_binders.function_names:
        _, cardinality = self._function_binders[token]
        current_node = Node(token, expansion_of="function")

        if quantifier_reached == 0:
          args = []
          for _ in range(cardinality):
            if len(execution_stack) == 0:
              raise RuntimeError(
                  "Stack is empty, check if the postfix_program is valid.")
            args.append(execution_stack.pop())
          args.reverse()

          for this_arg in args:
            current_node.add_child(this_arg)

        new_stack_top = current_node
      else:
        new_stack_top = Node(token, expansion_of="")
      execution_stack.append(new_stack_top)

    if len(execution_stack) not in [1, 2]:
      raise ValueError("Invalid Program.")
    return execution_stack

  def tree_to_postfix(self, list_tree):
    expression_tree = list_tree[0]

    if len(list_tree) == 2:
      quantifier = list_tree[1].op
      if len(list_tree[1]) != 0:
        raise ValueError
      quantifier_print = quantifier + r"x \in S "
    else:
      quantifier_print = ""

    def t2p(tree):
      if len(tree) == 0:
        return tree.op
      elif len(tree) == 1:
        return tree.op + "( " + t2p(tree._children[0]) + " )"
      elif len(tree) == 2:
        return tree.op + "(" + t2p(tree._children[0]) + ", " + t2p(
            tree._children[1]) + " )"
      else:
        raise ValueError("Maximum 2 children per node expected.")

    return quantifier_print + t2p(expression_tree)

  def postfix_to_prefix(self, postfix):
    tree = self.postfix_to_tree(postfix)
    prefix = self.tree_to_postfix(tree)
    return prefix


if __name__ == "__main__":
  properties_file = "concept_data/clevr_typed_fol_properties.json"
  grammar_expander = "concept_data/temp_data/v2_typed_simple_fol_clevr_typed_fol_properties.pkl"

  program_converter = PrefixPostfix(properties_file, grammar_expander)

  list_postfix = [
      "brown x color? = sphere x shape? = and lambda S. for-all=",
      "S locationY? x locationY? count= 2 > lambda S. exists=",
      "S color? x color? exists= S color? x color? for-all= and lambda S. for-all=",
      "x shape? cylinder = x color? brown = and lambda S. for-all=",
      "S color? red count= 1 > lambda S. for-all=",
      "S shape? x shape? for-all= S color? cyan for-all= and lambda S. for-all=",
      "S shape? cube for-all= S color? cyan for-all= and lambda S. exists=",
      "S color? x color? count= S locationY? x locationY? count= > lambda S. for-all=",
      "S locationX? x locationX? for-all= x shape? cylinder = and lambda S. exists=",
      "S size? 0.35 for-all= S material? rubber for-all= and lambda S. for-all=",
      "non-x-S color? brown for-all= non-x-S shape? sphere for-all= and lambda S. exists=",
      "S shape? sphere exists= red x color? = and lambda S. for-all=",
      "non-x-S locationX? 7 exists= non-x-S color? x color? exists= and lambda S. exists=",
      "blue x color? = S size? x size? for-all= and lambda S. for-all=",
      "non-x-S locationY? x locationY? for-all= S size? x size? exists= and lambda S. exists=",
      "S locationX? 1 for-all= S size? x size? for-all= and lambda S. exists=",
  ]
  for this_postfix in list_postfix:
    this_prefix = program_converter.postfix_to_prefix(this_postfix)
    print(this_postfix)
    print(this_prefix)

    print("______________")