"use client";

import {useEffect, useState} from 'react';
import {scanDocument} from '@/services/scanner';
import {extractCheckData} from '@/ai/flows/extract-check-data';
import {verifyAndCorrectCheckData} from '@/ai/flows/verify-and-correct-check-data';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {Label} from "@/components/ui/label";
import {Textarea} from "@/components/ui/textarea";
import {Alert, AlertDescription, AlertTitle} from "@/components/ui/alert";
import {Icons} from "@/components/icons";

export const Dashboard = () => {
  const [scanResult, setScanResult] = useState<string | null>(null);
  const [extractedData, setExtractedData] = useState<{
    amount: string;
    payee: string;
    date: string;
  } | null>(null);
  const [verificationResult, setVerificationResult] = useState<{
    isConsistent: boolean;
    correctedAmountInNumbers?: string;
    correctedAmountInWords?: string;
    flaggedIssues: string[];
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleScan = async () => {
    setIsLoading(true);
    setError(null);
    setExtractedData(null);
    setVerificationResult(null);

    try {
      const result = await scanDocument();
      setScanResult(result.base64ImageData);

      const extracted = await extractCheckData({base64ImageData: result.base64ImageData});
      setExtractedData(extracted);

      const verification = await verifyAndCorrectCheckData({
        amountInNumbers: extracted.amount,
        amountInWords: 'TODO',
        checkDate: extracted.date,
        payee: extracted.payee,
      });
      setVerificationResult(verification);
    } catch (e: any) {
      console.error('Error processing check:', e);
      setError(e.message || 'An error occurred while processing the check.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex p-6 min-h-screen bg-background rounded-lg shadow-md">
      <aside className="w-64 flex-none border-r pr-4 rounded-lg">
        <nav>
          <ul className="space-y-2">
            <li>
              <Button variant="outline" onClick={handleScan} disabled={isLoading} className="w-full">
                {isLoading ? (
                  <>
                    <Icons.spinner className="mr-2 h-4 w-4 animate-spin"/>
                    Scanning...
                  </>
                ) : (
                  <>
                    <Icons.search className="mr-2 h-4 w-4"/>
                    Scan Check
                  </>
                )}
              </Button>
            </li>
            <li>
              <a href="#" className="block py-2 hover:bg-accent rounded-md px-2">
                <Icons.home className="mr-2 h-4 w-4"/>
                Dashboard
              </a>
            </li>
            <li>
              <a href="#" className="block py-2 hover:bg-accent rounded-md px-2">
                <Icons.arrowRight className="mr-2 h-4 w-4"/>
                Transactions
              </a>
            </li>
            <li>
              <a href="#" className="block py-2 hover:bg-accent rounded-md px-2">
                <Icons.settings className="mr-2 h-4 w-4"/>
                Settings
              </a>
            </li>
          </ul>
        </nav>
      </aside>

      <main className="flex-1 p-4">
        <h1 className="text-3xl font-bold mb-4">Check Processing Dashboard</h1>

        {error && (
          <Alert variant="destructive">
            <Icons.close className="h-4 w-4"/>
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="shadow-md rounded-lg">
            <CardHeader>
              <CardTitle>Scanned Check Image</CardTitle>
              <CardDescription>Review the scanned check image.</CardDescription>
            </CardHeader>
            <CardContent>
              {scanResult ? (
                <img src={`data:image/png;base64,${scanResult}`} alt="Scanned Check"
                     className="max-w-full rounded-md shadow-md"/>
              ) : (
                <div className="flex items-center justify-center h-48 bg-muted rounded-md">
                  {isLoading ? (
                    <Icons.spinner className="h-6 w-6 animate-spin text-muted-foreground"/>
                  ) : (
                    <span className="text-muted-foreground">No check scanned yet.</span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="shadow-md rounded-lg">
            <CardHeader>
              <CardTitle>Extracted Data</CardTitle>
              <CardDescription>Manually extracted data can be modified below.</CardDescription>
            </CardHeader>
            <CardContent>
              {extractedData ? (
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="amount">Amount</Label>
                    <Input id="amount" defaultValue={extractedData?.amount || ''}
                           className="shadow-sm focus-visible:ring-1 focus-visible:ring-ring"/>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="payee">Payee</Label>
                    <Input id="payee" defaultValue={extractedData?.payee || ''}
                           className="shadow-sm focus-visible:ring-1 focus-visible:ring-ring"/>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="date">Date</Label>
                    <Input id="date" defaultValue={extractedData?.date || ''}
                           className="shadow-sm focus-visible:ring-1 focus-visible:ring-ring"/>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="notes">Notes</Label>
                    <Textarea id="notes" className="shadow-sm focus-visible:ring-1 focus-visible:ring-ring"/>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-48 bg-muted rounded-md">
                  {isLoading ? (
                    <Icons.spinner className="h-6 w-6 animate-spin text-muted-foreground"/>
                  ) : (
                    <span className="text-muted-foreground">No data extracted yet.</span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {verificationResult && (
          <Card className="mt-4 shadow-md rounded-lg">
            <CardHeader>
              <CardTitle>Verification Results</CardTitle>
              <CardDescription>Review the verification results for inconsistencies.</CardDescription>
            </CardHeader>
            <CardContent>
              <p>Is Consistent: {verificationResult.isConsistent ? 'Yes' : 'No'}</p>
              {verificationResult.correctedAmountInNumbers && (
                <p>Corrected Amount (Numbers): {verificationResult.correctedAmountInNumbers}</p>
              )}
              {verificationResult.correctedAmountInWords && (
                <p>Corrected Amount (Words): {verificationResult.correctedAmountInWords}</p>
              )}
              {verificationResult.flaggedIssues.length > 0 && (
                <div>
                  <p>Flagged Issues:</p>
                  <ul>
                    {verificationResult.flaggedIssues.map((issue, index) => (
                      <li key={index}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};
