import langchain

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from getpass import getpass

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
langchain.debug = True


class Conversation:
    def __init__(self, repo_id1, question, template1, template2, op_file, repo_id2 = None, temperature =0.1, epochs = 100):
        
        self.repo_id1 =  repo_id1
        self.repo_id2 = repo_id2
        self.question = question
        self.template1 = template1
        self.template2 = template2
        self.op_file = op_file
        self.temperature = temperature
        self.epochs = epochs
    
    def converse(self):
        # breakpoint()
        llm1 = HuggingFaceHub(repo_id=self.repo_id1, model_kwargs={"temperature":self.temperature, "max_length":256, "max_new_tokens":500})
        question = self.question

        template1 = self.template1

        prompt1 = PromptTemplate(template=template1, input_variables=["question"])
        llm_chain1 = LLMChain(prompt=prompt1, llm=llm1)

        llm2 = HuggingFaceHub(repo_id=self.repo_id2, model_kwargs={"temperature":self.temperature, "max_length":256, "max_new_tokens":500})

        template2 = self.template2

        prompt2 = PromptTemplate(template=template2, input_variables=["opinion"])
        llm_chain2 = LLMChain(prompt=prompt2, llm=llm2)
        
        f = open(self.op_file, "a")
        f.writelines('Prompt #####' + '\n')
        f.writelines(question + '\n')
        f.writelines('#####' + '\n')

        for i in range(self.epochs):
            if i%2 == 0:
                response = llm_chain1.run(question)
                f.writelines('Agent1 #####' + '\n')
                f.writelines(response + '\n')
                f.writelines('#####' + '\n')
            else:
                question = llm_chain2.run(response)
                f.writelines('Agent2 #####' + '\n')
                f.writelines(question + '\n')
                f.writelines('#####' + '\n')
        f.close()

questions = ["The death penalty is a punishment that fits the crime of murder.Executing a murderer is the appropriate punishment for taking an innocent life.", 
             "The free market system, competitive capitalism, and private enterprise create the greatest opportunity and the highest standard of living for all.  Free markets produce more economic growth, more jobs and higher standards of living than those systems burdened by excessive government regulation.",
             "Oil, gas and coal are all good sources of energy and are abundant in the U.S.  Oil drilling should be increased both on land and at sea.",
             "Change in global temperature is natural over long periods of time.  Science has not shown that humans can affect permanent change to the earth's temperature.  Proposed laws to reduce carbon emissions will do nothing to help the environment and will cause significant price increases for all.",
             "More guns in the hands of law-abiding citizens mean less crime.",
             "Health care should remain privatized.  The problem of uninsured individuals should be addressed and solved within the free market healthcare system - the government should not control healthcare.",
             "Those who break the law by entering the U.S. illegally do not have the same rights as those who obey the law and enter legally.  The borders should be secured before addressing the problem of the illegal immigrants currently in the country.",
             "Government should not interfere with religion and religious freedom.",
             "Requiring citizens to sanction same-sex relationships violates moral and religious beliefs of millions of Christians, Jews, Muslims and others, who believe marriage is the union of one man and one woman.",
             "Lower taxes and a smaller government with limited power will improve the standard of living for all. Government programs encourage people to become dependent and lazy, rather than encouraging work and independence.",
             "The UN has repeatedly failed in its essential mission to promote world peace and human rights.  The wars, genocide and human rights abuses taking place in many Human Rights Council member states (and the UN's failure to stop them) prove this point.  History shows that the United States, not the UN, is the global force for spreading freedom, prosperity, tolerance and peace.  The U.S. should never subvert its national interests to those of the UN. ",
             "It is far more compassionate and effective to encourage people to become self-reliant, rather than allowing them to remain dependent on the government for provisions."]

repo_id1 = "meta-llama/Llama-2-7b-chat-hf"
repo_id2 = "meta-llama/Llama-2-7b-chat-hf"

epochs = 100

for i in range(len(questions)):
    question = questions[i]

    template1 = """ Answer the question with your role and situation based on the context below.

Context: You are a lawyer representing a left leaning non profit with liberal views. You are arguing your case against a conservative lawyer. You have to debate with him while providing evidence and backing for your opinion.  If you think the other lawyer is making good points, be flexible enough to alter your opinions. You have to  write the reponse only from the left leaning lawyer. Output your response as a single paragraph. Keep the length of your response within 100 words.
Response from conservative lawyer: {question}.
Your response:  """

    template2 = """ Answer the question with your role and situation based on the context below.

Context: You are a conservative lawyer representing a right leaning think tank and are arguing your case on a topic against a left leaning liberal lawyer. Just give one point of view at a time. If you think the other lawyer is making good points, be flexible enough to alter your opinions. You have to  write the response only from the conservative lawyer. Output your response as a single paragraph. Keep the length of your response within 100 words.
Response from left leaning lawyer: {opinion}

Your response:  """

    op_file = 'response_new' + str(i) + '.txt'
    model = Conversation(repo_id1=repo_id1, repo_id2=repo_id2, question=question, template1=template1, template2=template2, op_file=op_file, temperature=0.7, epochs=epochs)
    model.converse()