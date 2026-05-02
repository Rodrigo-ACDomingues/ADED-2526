# ADED-2526
Repositório criado para a disciplina de Análise de Dados de Elevado Desempenho, ano letivo 2025/2026

Os slurm job scripts (.sh) para a recolha das métricas são os mesmos que foram apresentados durante a aula prática acerca do
guião 4, para retirar as medições foram se feitas alterações apenas ao número de workers e de threads feito através de passar
como argumento as variáveis de NUM_WORKERS= e CPU_THREADS_PER_WORKER=, foi recolhido o output do prompt (para S, M e L, as 3 medições
de cada um) juntamente com os valores indicados no ficheiro de head.err, com estes dados é populado varios .txt files para cada uma
das configurações do modelo.

Os modelos usados foram o meta-llama providenciado durante a aula, o TinyLlama accesivél neste endereço do hugging face: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF, e por fim o Gemma2 accessivél neste endereço: https://huggingface.co/BafS/gemma-2-2b-it-Q4_K_M-GGUF

Para executar o script que compila e faz o plot dos dados segundo os vários gráficos basta correr o seguinte comando na consola no local
onde se encontra o script (neste caso está dentro de ADED_PROJ):

- python .\plotting.py .\pasta_com_dados_que_se_pretende_compilar --out_dir .\pasta_para_onde_devem_ir_os_resultados