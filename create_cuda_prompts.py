#!/usr/bin/env python3
"""
Script para extrair apenas prompts CUDA do arquivo de prompts completo
"""
import json
import argparse

def filter_cuda_prompts(input_file, output_file):
    """Filtra apenas prompts CUDA do arquivo de entrada"""
    
    # Carregar prompts
    with open(input_file, 'r') as f:
        all_prompts = json.load(f)
    
    # Filtrar apenas CUDA
    cuda_prompts = [
        prompt for prompt in all_prompts 
        if prompt.get('parallelism_model') == 'cuda'
    ]
    
    print(f"Total de prompts: {len(all_prompts)}")
    print(f"Prompts CUDA: {len(cuda_prompts)}")
    
    # Salvar prompts CUDA
    with open(output_file, 'w') as f:
        json.dump(cuda_prompts, f, indent=2)
    
    print(f"Prompts CUDA salvos em: {output_file}")
    
    # Mostrar alguns exemplos
    print("\nExemplos de problemas CUDA:")
    for i, prompt in enumerate(cuda_prompts[:5]):
        print(f"{i+1}. {prompt['name']} ({prompt['problem_type']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrair prompts CUDA')
    parser.add_argument('--input', default='prompts/generation-prompts.json', 
                       help='Arquivo de prompts de entrada')
    parser.add_argument('--output', default='prompts/cuda-only-prompts.json',
                       help='Arquivo de saída com apenas prompts CUDA')
    
    args = parser.parse_args()
    filter_cuda_prompts(args.input, args.output)