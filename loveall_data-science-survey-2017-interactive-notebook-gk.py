#!/usr/bin/env python
# coding: utf-8



import pandas
from collections import Counter
import numpy
from nltk.corpus import stopwords
import re
survey_questions=pandas.read_csv('../input/schema.csv')
freeform_response=pandas.read_csv('../input/freeformResponses.csv')
Questions_list=[]
AnswerClouds_list=[]
for a in freeform_response.columns.values:
    tokenized_sents = [re.findall("[\w']+ [\w']+", i) for i in freeform_response[freeform_response[a].notnull()][a].unique()]
    lower_cased=[x.lower() for x in [item for sublist in tokenized_sents for item in sublist]]
    filtered= [w for w in lower_cased if not w in stopwords.words('english')]
    word_count=Counter([item for item in filtered])
    # stop = stopwords.words('english')
    Questions_list.append(a)
    AnswerClouds_list.append(word_count)
freeform_response_transformed=pandas.DataFrame(AnswerClouds_list)
freeform_response_transformed.insert(0, 'Questions', Questions_list)
freeform_response_transformed=freeform_response_transformed.set_index(['Questions']).unstack().reset_index()
freeform_response_transformed.columns=['Word','Questions','Word Count']
freeform_response_transformed=freeform_response_transformed[['Questions','Word','Word Count']]
freeform_response_transformed=freeform_response_transformed[numpy.isfinite(freeform_response_transformed['Word Count'])]
freeform_response_transformed=freeform_response_transformed.sort_values(['Questions', 'Word Count'], ascending=[True, False])
freeform_response_transformed=pandas.merge(freeform_response_transformed, survey_questions, how='inner', left_on=['Questions'], right_on=['Column'])
freeform_response_transformed.columns=['QuestionType','Word','Word Count','QuestionType2','Question','AskedFrom']
freeform_response_transformed=freeform_response_transformed[['AskedFrom','QuestionType','Question','Word','Word Count']]
freeform_response_transformed




get_ipython().run_cell_magic('HTML', '', "\n<div class='tableauPlaceholder' id='viz1510207613824' style='position: relative'><noscript><a href='#'><img alt='Data Science Survey 2017 Dashboard ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ka&#47;KaggleDataScienceSurvey2017&#47;DataScienceSurvey2017Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='KaggleDataScienceSurvey2017&#47;DataScienceSurvey2017Dashboard' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ka&#47;KaggleDataScienceSurvey2017&#47;DataScienceSurvey2017Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1510207613824');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='800px';vizElement.style.height='1250px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1510215951375' style='position: relative'><noscript><a href='https:&#47;&#47;www.mesumrazahemani.wordpress.com'><img alt='Kaggle Data Science Survey 2017 Insights ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ka&#47;KaggleDataScienceSurvey2017&#47;KaggleDataScienceSurvey2017Insights&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='KaggleDataScienceSurvey2017&#47;KaggleDataScienceSurvey2017Insights' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ka&#47;KaggleDataScienceSurvey2017&#47;KaggleDataScienceSurvey2017Insights&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1510215951375');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='800px';vizElement.style.height='850px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")

