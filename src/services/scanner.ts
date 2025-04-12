/**
 * Represents the result of a scan.
 */
export interface ScanResult {
  /**
   * The base64 encoded image data.
   */
  base64ImageData: string;
}

/**
 * Asynchronously scans a document by allowing the user to upload an image.
 *
 * @returns A promise that resolves to a ScanResult object containing the scanned image data.
 */
export async function scanDocument(): Promise<ScanResult> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';

    input.onchange = async (event: Event) => {
      const target = event.target as HTMLInputElement;
      if (target.files && target.files.length > 0) {
        const file = target.files[0];
        const reader = new FileReader();

        reader.onload = (e: any) => {
          const base64ImageData = e.target.result as string;
          resolve({ base64ImageData });
        };

        reader.onerror = () => {
          reject(new Error('Error reading file'));
        };

        reader.readAsDataURL(file);
      } else {
        reject(new Error('No file selected'));
      }
    };

    input.click();
  });
}
