'use server';
/**
 * @fileOverview This file defines a Genkit flow for extracting data from a check image.
 *
 * It includes the following:
 * - extractCheckData - A function that extracts data from a check image.
 * - ExtractCheckDataInput - The input type for the extractCheckData function.
 * - ExtractCheckDataOutput - The return type for the extractCheckData function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const ExtractCheckDataInputSchema = z.object({
  base64ImageData: z.string().describe('The base64 encoded image data of the check.'),
});
export type ExtractCheckDataInput = z.infer<typeof ExtractCheckDataInputSchema>;

const ExtractCheckDataOutputSchema = z.object({
  amount: z.string().describe('The amount on the check.'),
  payee: z.string().describe('The payee on the check.'),
  date: z.string().describe('The date on the check.'),
});
export type ExtractCheckDataOutput = z.infer<typeof ExtractCheckDataOutputSchema>;

export async function extractCheckData(input: ExtractCheckDataInput): Promise<ExtractCheckDataOutput> {
  return extractCheckDataFlow(input);
}

const prompt = ai.definePrompt({
  name: 'extractCheckDataPrompt',
  input: {
    schema: z.object({
      base64ImageData: z.string().describe('The base64 encoded image data of the check.'),
    }),
  },
  output: {
    schema: z.object({
      amount: z.string().describe('The amount on the check.'),
      payee: z.string().describe('The payee on the check.'),
      date: z.string().describe('The date on the check.'),
    }),
  },
  prompt: `You are an expert in extracting data from checks.  Given the following check image, extract the amount, payee, and date.

Check Image: {{media url=base64ImageData mimeType=\'image/png\'}}

Ensure that the outputted JSON is valid and that the field values are correct.`, // Ensure valid JSON
});

const extractCheckDataFlow = ai.defineFlow<
  typeof ExtractCheckDataInputSchema,
  typeof ExtractCheckDataOutputSchema
>({
  name: 'extractCheckDataFlow',
  inputSchema: ExtractCheckDataInputSchema,
  outputSchema: ExtractCheckDataOutputSchema,
},
async input => {
  const {output} = await prompt(input);
  return output!;
});

