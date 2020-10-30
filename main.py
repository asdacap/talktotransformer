import prompt_toolkit
import argparse
import yaml
import sys
import os
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from abc import ABC, abstractmethod
from typing import List, Dict

psess = prompt_toolkit.PromptSession(erase_when_done = True)

class Conversation():
    actor: str
    text: str
    max_response_length = 100
    def __init__(self, actor: str, text: str):
        self.actor = actor
        self.text = text

    def as_dict(self):
        return {
            'actor': self.actor,
            'text': self.text
        }

class DialogContext():
    context: str
    conversations: List[Conversation]

    def __init__(self, context: str, conversations: List[Conversation] = []):
        self.context = context
        self.conversations = [Conversation(**x) for x in conversations]

    def as_dict(self):
        return {
            'context': self.context,
            'conversations': [x.as_dict() for x in self.conversations]
        }

class Actor(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_answer(self, context: DialogContext):
        pass

class HumanActor(Actor):
    name: str

    def __init__(self, name: str):
        super(HumanActor, self).__init__(name)

    def get_answer(self, context: DialogContext):
        return psess.prompt(self.name + ": ")

class AIActor(Actor):
    max_response_length: int
    max_conversation_in_context: int
    generate_args: Dict

    def __init__(
            self,
            name: str,
            model_name: str,
            max_response_length = 100,
            max_conversation_in_context = 10,
            generate_args = {}
    ):
        super(AIActor, self).__init__(name)
        from transformers import AutoTokenizer

        self.max_response_length = max_response_length
        self.max_conversation_in_context = max_conversation_in_context

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.newline = "\n"
        self.eos_token_id = self.tokenizer(self.newline + "a")["input_ids"][0]
        #eos_token_id = 198

        self.generate_args = {
            "repetition_penalty": 1.05,
            **generate_args,
        }

    def generate_text_for_generation(self, context: DialogContext):
        text = context.context + self.newline
        for conversation in context.conversations[-self.max_conversation_in_context:]:
            text = text + conversation.actor + ": " + conversation.text + self.newline
        return text + self.name + ": "

    def show_loading_prompt(self):
        sys.stdout.write(self.name + " is thinking...")
        sys.stdout.flush()

    def hide_loading_prompt(self):
        message = self.name + " is thinking..."
        sys.stdout.write("\r"+(" "*len(message))+"\r")
        sys.stdout.flush()

    def get_answer(self, context: DialogContext):
        self.show_loading_prompt()

        current_text = self.generate_text_for_generation(context)
        tokenized = self.tokenizer(current_text, return_tensors='tf')
        input_ids = tokenized['input_ids']
        tokenized_length = input_ids.shape[1]
        output_sequences = self.model.generate(input_ids=input_ids,
                                               max_length=tokenized_length + self.max_response_length,
                                               pad_token_id=self.eos_token_id,
                                               eos_token_id=self.eos_token_id,
                                               **self.generate_args)

        response = self.tokenizer.decode(output_sequences[0].numpy().tolist()[tokenized_length:])

        self.hide_loading_prompt()
        return response.strip()

class CausalLMActor(AIActor):
    def __init__(self, name: str, model_name: str, **kwargs):
        super(CausalLMActor, self).__init__(name, model_name, **kwargs)
        from transformers import TFAutoModelForCausalLM

        self.model = TFAutoModelForCausalLM.from_pretrained(model_name)

class MaskedLMActor(AIActor):
    def __init__(self, name: str, model_name: str, **kwargs):
        super(MaskedLMActor, self).__init__(name, model_name, **kwargs)
        from transformers import TFAutoModelForMaskedLM

        self.model = TFAutoModelForMaskedLM.from_pretrained(model_name)

class DialogController:

    state: DialogContext
    session_file: str
    config: Dict
    actors: List[Actor]
    current_actor_idx = -1

    def __init__(self, config_file: str, session_file: str):
        self.session_file = session_file
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.CLoader)

        self.state = DialogContext(**config['state'])
        if os.path.exists(session_file):
            print("Session file found. Resuming session.")
            with open(session_file) as f:
                state = yaml.load(f, Loader=yaml.CLoader)
                self.state = DialogContext(**state)

        self.actors = [self.load_actor(x) for x in config['actors']]

        if len(self.state.conversations) > 0:
            self.current_actor_idx = [x.name for x in self.actors].index(self.state.conversations[-1].actor)

    def load_actor(self, actor_config: Dict):
        type = actor_config.pop("type")

        if (type == "Human"):
            return HumanActor(**actor_config)
        elif (type == "CausalLM"):
            return CausalLMActor(**actor_config)
        elif (type == "MaskedLM"):
            return MaskedLMActor(**actor_config)
        else:
            raise Exception("Unknown actor type " + type)

    def save_session(self):
        with open(self.session_file, 'w') as f:
            yaml.safe_dump(self.state.as_dict(), f)

    def run(self):
        print()
        print()
        print_formatted_text(FormattedText([("#00ff00", self.state.context)]))
        for conversation in self.state.conversations:
            print_formatted_text(FormattedText([
                ("#0000ff", conversation.actor + ": "),
                ("#cccccc", conversation.text),
            ]))

        while True:
            self.current_actor_idx = (self.current_actor_idx + 1)%len(self.actors)
            actor = self.actors[self.current_actor_idx]
            text = actor.get_answer(self.state)
            self.state.conversations.append(Conversation(actor.name, text))
            print_formatted_text(FormattedText([
                ("#0000ff", actor.name + ": "),
                ("#cccccc", text),
            ]))

            self.save_session()

def load_from_yaml(file: str):
    basename = os.path.basename(file)
    session_file = file
    dir = os.path.dirname(file)
    if not os.path.splitext(basename)[0].endswith('.session'):
        session_file = os.path.join(dir, os.path.splitext(basename)[0] + '.session.yaml')

    return DialogController(file, session_file)

parser = argparse.ArgumentParser(description='Run a conversation between you and an AI')
parser.add_argument('scenario', type=str,
                    help='The yaml scenario file to run')

args = parser.parse_args()
if args.scenario:
    load_from_yaml(args.scenario).run()

